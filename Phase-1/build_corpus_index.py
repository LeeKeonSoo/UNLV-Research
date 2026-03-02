"""
Step 2a: Build Corpus Index for Redundancy Detection

Builds a searchable index over all configured datasets once,
so Step 2 can efficiently compute redundancy metrics without O(n²) comparisons.

Output:
  outputs/corpus_index.pkl
  - exact_hash_counts: Counter of MD5 hashes for exact-duplicate detection
  - doc_hash_by_id: {doc_id: text_hash} mapping
  - doc_texts: {doc_id: raw chunk text} for n-gram overlap checks (or SQLite backend)
  - lsh: MinHash LSH index for near-duplicate detection
  - minhashes: {doc_id: MinHash} for Jaccard similarity
  - corpus_matrix: sparse TF-IDF matrix for semantic similarity
  - vectorizer: fitted TF-IDF vectorizer
  - doc_ids: ordered list of document IDs
"""

import os
import hashlib
import json
import pickle
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# ==============================================================================
# Config
# ==============================================================================

OUTPUT_PATH      = "outputs/corpus_index.pkl"
DATASETS_CONFIG_PATH = os.getenv("PHASE1_DATASETS_CONFIG", "datasets_config.json")
PROJECT_DIR = Path(__file__).resolve().parent

# For LSH: lower threshold = more candidate pairs (more recall, less precision)
def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


LSH_THRESHOLD = _env_float("PHASE1_LSH_THRESHOLD", 0.5)
NUM_PERM = _env_int("PHASE1_NUM_PERM", 128)   # MinHash permutations (higher = more accurate)

# TF-IDF for semantic similarity
TFIDF_MAX_FEATURES = _env_int("PHASE1_TFIDF_MAX_FEATURES", 5000)  # Larger vocab for semantic comparison
BUILD_TFIDF_MATRIX = _env_bool("PHASE1_BUILD_TFIDF_MATRIX", False)
STORE_DOC_TEXTS = _env_bool("PHASE1_STORE_DOC_TEXTS", True)
ENABLE_MINHASH = _env_bool("PHASE1_ENABLE_MINHASH", True)
DOC_TEXT_BACKEND = os.getenv("PHASE1_DOC_TEXT_BACKEND", "sqlite").strip().lower()
DOC_TEXT_DB_PATH = os.getenv("PHASE1_DOC_TEXT_DB_PATH", "outputs/corpus_texts.sqlite")
DOC_TEXT_INSERT_BATCH = max(256, _env_int("PHASE1_DOC_TEXT_INSERT_BATCH", 2000))

# Chunk size mirror of compute_metrics.py
CHUNK_SIZE = 200
MIN_CHUNK_WORDS = 20
INDEX_MAX_BATCHES = _env_int("PHASE1_INDEX_MAX_BATCHES", 0) or None


# ==============================================================================
# Text chunking (mirrors compute_metrics.py)
# ==============================================================================

import re

def chunk_text(text: str, chunk_size: int = 200):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= chunk_size:
            if len(words) >= MIN_CHUNK_WORDS:
                chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            cur, cur_len = [], 0
            for sent in sentences:
                sw = len(sent.split())
                if cur_len + sw > chunk_size and cur:
                    joined = " ".join(cur)
                    if len(joined.split()) >= MIN_CHUNK_WORDS:
                        chunks.append(joined)
                    cur, cur_len = [sent], sw
                else:
                    cur.append(sent)
                    cur_len += sw
            if cur:
                joined = " ".join(cur)
                if len(joined.split()) >= MIN_CHUNK_WORDS:
                    chunks.append(joined)
    return chunks


def _slugify_dataset_name(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "dataset"


def _default_dataset_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "khan_academy",
            "format": "json_list",
            "source": "khan_k12_concepts/all_k12_concepts.json",
            "text_field": "content",
            "id_fields": ["doc_id", "url"],
            "min_text_chars": 50,
        },
        {
            "name": "tiny_textbooks",
            "format": "json_batch_dir",
            "source": "tiny_textbooks_raw",
            "batch_glob": "batch_*.json",
            "text_field": "text",
            "id_fields": ["id"],
            "min_text_chars": 100,
        },
    ]


def _normalize_dataset_spec(raw: Dict[str, Any], idx: int) -> Dict[str, Any]:
    name = _slugify_dataset_name(raw.get("name", f"dataset_{idx+1}"))
    fmt = str(raw.get("format", "json_list")).strip().lower()
    source = str(raw.get("source", "")).strip()
    if not source:
        raise ValueError(f"datasets[{idx}].source is required")
    if fmt not in {"json_list", "json_batch_dir"}:
        raise ValueError(
            f"datasets[{idx}].format must be one of ['json_list','json_batch_dir']"
        )
    id_fields = raw.get("id_fields")
    if isinstance(id_fields, str):
        id_fields = [id_fields]
    if not isinstance(id_fields, list) or not id_fields:
        id_fields = ["doc_id", "id", "url"]
    return {
        "name": name,
        "format": fmt,
        "source": source,
        "batch_glob": str(raw.get("batch_glob") or "batch_*.json"),
        "text_field": str(raw.get("text_field") or "text"),
        "id_fields": [str(x) for x in id_fields],
        "min_text_chars": int(raw.get("min_text_chars", 50)),
    }


def _load_dataset_specs(config_path: str = DATASETS_CONFIG_PATH) -> List[Dict[str, Any]]:
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = PROJECT_DIR / cfg
    if not cfg.exists():
        return _default_dataset_specs()
    with cfg.open("r", encoding="utf-8", errors="replace") as f:
        payload = json.load(f)
    raw_specs = payload.get("datasets", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"Invalid dataset config at {cfg}. Expected non-empty datasets list.")
    return [_normalize_dataset_spec(raw, i) for i, raw in enumerate(raw_specs)]


def _stable_doc_base_id(
    doc: Dict[str, Any],
    doc_idx: int,
    dataset_name: str,
    id_fields: List[str],
    text_field: str,
) -> str:
    for key in id_fields:
        raw = str(doc.get(key) or "").strip()
        if raw and raw.lower() != "unknown":
            return raw
    title = str(doc.get("title") or "").strip()
    text = str(doc.get(text_field) or "").strip()
    seed = f"{title}\n{text[:500]}"
    suffix = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12] if seed.strip() else "na"
    return f"missing_{dataset_name}_id_{doc_idx}_{suffix}"


def _dedupe_with_counter(base_id: str, seen: Dict[str, int]) -> str:
    n = seen.get(base_id, 0) + 1
    seen[base_id] = n
    if n == 1:
        return base_id
    return f"{base_id}__dup{n}"


def _open_text_db(path: str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, text TEXT NOT NULL)")
    conn.commit()
    return conn


# ==============================================================================
# Document loading
# ==============================================================================

def iter_documents(dataset_specs: List[Dict[str, Any]]):
    """Yield (doc_id, text) for every chunk across configured datasets."""
    for spec in dataset_specs:
        name = spec["name"]
        source_path = Path(spec["source"])
        text_field = spec["text_field"]
        id_fields = spec["id_fields"]
        min_text_chars = int(spec["min_text_chars"])

        if spec["format"] == "json_list":
            if not source_path.exists():
                continue
            print(f"Loading {name} (json_list)...")
            with source_path.open("r", encoding="utf-8", errors="replace") as f:
                docs = json.load(f)
            if not isinstance(docs, list):
                continue
            seen_doc_ids: Dict[str, int] = {}
            for doc_idx, doc in enumerate(tqdm(docs, desc=f"{name} chunks")):
                text = str(doc.get(text_field, ""))
                if len(text.strip()) < min_text_chars:
                    continue
                base_id = _dedupe_with_counter(
                    _stable_doc_base_id(doc, doc_idx, name, id_fields, text_field),
                    seen_doc_ids,
                )
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                    yield f"{name}::{base_id}::{i}", chunk
            continue

        if not source_path.exists():
            continue
        batch_files = sorted(source_path.glob(spec["batch_glob"]))
        if INDEX_MAX_BATCHES:
            batch_files = batch_files[:INDEX_MAX_BATCHES]
            print(f"  (index mode: first {INDEX_MAX_BATCHES} batches only)")
        print(f"\nLoading {name} ({len(batch_files)} batches)...")
        for batch_file in tqdm(batch_files, desc=f"{name} batches"):
            with batch_file.open("r", encoding="utf-8", errors="replace") as f:
                batch = json.load(f)
            if not isinstance(batch, list):
                continue
            seen_batch_ids: Dict[str, int] = {}
            for doc_idx, doc in enumerate(batch):
                text = str(doc.get(text_field, ""))
                if len(text.strip()) < min_text_chars:
                    continue
                doc_id = _dedupe_with_counter(
                    _stable_doc_base_id(doc, doc_idx, name, id_fields, text_field),
                    seen_batch_ids,
                )
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                    yield f"{name}::{batch_file.name}::{doc_id}::{i}", chunk


# ==============================================================================
# Index building
# ==============================================================================

def build_corpus_index():
    Path("outputs").mkdir(exist_ok=True)
    dataset_specs = _load_dataset_specs()

    print("="*60)
    print("BUILDING CORPUS INDEX")
    print("="*60)
    print(f"Datasets config: {DATASETS_CONFIG_PATH}")
    for spec in dataset_specs:
        print(f"  - {spec['name']} ({spec['format']}) source={spec['source']}")

    doc_ids   = []
    texts     = [] if BUILD_TFIDF_MATRIX else None
    exact_hash_counts = Counter()
    doc_hash_by_id = {}
    doc_texts = {}
    minhashes = {}
    text_db_conn = None
    text_db_rows = []
    lsh = None if not ENABLE_MINHASH else MinHashLSH(
        threshold=LSH_THRESHOLD, num_perm=NUM_PERM
    )

    if STORE_DOC_TEXTS and DOC_TEXT_BACKEND == "sqlite":
        text_db_conn = _open_text_db(DOC_TEXT_DB_PATH)
    elif STORE_DOC_TEXTS and DOC_TEXT_BACKEND != "memory":
        print(
            f"⚠ Unsupported PHASE1_DOC_TEXT_BACKEND='{DOC_TEXT_BACKEND}'. "
            "Falling back to in-memory doc_texts."
        )

    print("\nPass 1: collecting chunks, exact hashes, MinHash signatures...")
    for doc_id, text in iter_documents(dataset_specs):
        doc_ids.append(doc_id)
        if texts is not None:
            texts.append(text)

        # Exact duplicate hash/counts + doc lookup
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        exact_hash_counts[text_hash] += 1
        doc_hash_by_id[doc_id] = text_hash
        if STORE_DOC_TEXTS:
            if text_db_conn is not None:
                text_db_rows.append((doc_id, text))
                if len(text_db_rows) >= DOC_TEXT_INSERT_BATCH:
                    text_db_conn.executemany(
                        "INSERT OR REPLACE INTO docs(doc_id, text) VALUES (?, ?)",
                        text_db_rows,
                    )
                    text_db_conn.commit()
                    text_db_rows.clear()
            else:
                doc_texts[doc_id] = text

        # MinHash
        if ENABLE_MINHASH:
            mh = MinHash(num_perm=NUM_PERM)
            # MinHash models set similarity; repeated tokens do not add information.
            for word in set(text.lower().split()):
                mh.update(word.encode("utf-8"))
            minhashes[doc_id] = mh

            # Insert into LSH (silently skip if collision)
            try:
                lsh.insert(doc_id, mh)
            except ValueError:
                pass  # duplicate key – already indexed

    if text_db_conn is not None:
        if text_db_rows:
            text_db_conn.executemany(
                "INSERT OR REPLACE INTO docs(doc_id, text) VALUES (?, ?)",
                text_db_rows,
            )
            text_db_conn.commit()
            text_db_rows.clear()
        text_db_conn.close()

    print(f"\n✓ Collected {len(doc_ids):,} chunks total")

    print("\nPass 2: building TF-IDF matrix for semantic similarity...")
    vectorizer = None
    corpus_matrix = None
    if BUILD_TFIDF_MATRIX and texts is not None:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=2,
            dtype=np.float32,
        )
        corpus_matrix = vectorizer.fit_transform(texts)
        print(f"✓ TF-IDF matrix: {corpus_matrix.shape}")
    else:
        print("⚠️ Skipping TF-IDF matrix build (PHASE1_BUILD_TFIDF_MATRIX=0). "
              "Semantic redundancy will use lexical fallback in Step 2.")

    exact_hashes = set(exact_hash_counts.keys())  # backward compatibility

    index = {
        "exact_hashes":      exact_hashes,  # legacy key
        "exact_hash_counts": exact_hash_counts,
        "doc_hash_by_id":    doc_hash_by_id,
        "doc_texts":         doc_texts,
        "lsh":               lsh,
        "minhashes":         minhashes,
        "corpus_matrix":     corpus_matrix,
        "vectorizer":        vectorizer,
        "doc_ids":           doc_ids,
        "doc_texts_backend": (
            DOC_TEXT_BACKEND if STORE_DOC_TEXTS else "none"
        ),
        "doc_texts_db_path": (
            DOC_TEXT_DB_PATH if STORE_DOC_TEXTS and DOC_TEXT_BACKEND == "sqlite" else None
        ),
        "BUILD_TFIDF_MATRIX": BUILD_TFIDF_MATRIX,
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(index, f)

    print(f"\n✓ Saved corpus index to {OUTPUT_PATH}")
    duplicate_chunks = sum(c for c in exact_hash_counts.values() if c > 1)
    duplicate_hashes = sum(1 for c in exact_hash_counts.values() if c > 1)
    print(f"  Chunks indexed      : {len(doc_ids):,}")
    print(f"  Unique hashes       : {len(exact_hashes):,}")
    print(f"  Duplicate hash bins : {duplicate_hashes:,}")
    print(f"  Duplicate chunks    : {duplicate_chunks:,}")
    tfidf_vocab = len(vectorizer.vocabulary_) if vectorizer is not None else 0
    print(f"  TF-IDF vocab        : {tfidf_vocab:,}")


if __name__ == "__main__":
    build_corpus_index()
