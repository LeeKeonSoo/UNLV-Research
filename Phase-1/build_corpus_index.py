"""
Step 2a: Build Corpus Index for Redundancy Detection

Builds a searchable index over all documents (Khan + Tiny-Textbooks) once,
so Step 2 can efficiently compute redundancy metrics without O(n²) comparisons.

Output:
  outputs/corpus_index.pkl
  - exact_hash_counts: Counter of MD5 hashes for exact-duplicate detection
  - doc_hash_by_id: {doc_id: text_hash} mapping
  - doc_texts: {doc_id: raw chunk text} for n-gram overlap checks
  - lsh: MinHash LSH index for near-duplicate detection
  - minhashes: {doc_id: MinHash} for Jaccard similarity
  - corpus_matrix: sparse TF-IDF matrix for semantic similarity
  - vectorizer: fitted TF-IDF vectorizer
  - doc_ids: ordered list of document IDs
"""

import hashlib
import json
import pickle
import sqlite3
from collections import Counter
from pathlib import Path
import os
from typing import Dict

import numpy as np
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# ==============================================================================
# Config
# ==============================================================================

KHAN_DATA_PATH   = "khan_k12_concepts/all_k12_concepts.json"
TINY_DATA_DIR    = "tiny_textbooks_raw"
OUTPUT_PATH      = "outputs/corpus_index.pkl"

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


def _stable_khan_doc_base_id(doc: Dict, doc_idx: int) -> str:
    raw = str(doc.get("doc_id") or doc.get("url") or "").strip()
    if raw and raw.lower() != "unknown":
        return raw
    title = str(doc.get("title") or "").strip()
    text = str(doc.get("content") or "").strip()
    seed = f"{title}\n{text[:500]}"
    suffix = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12]
    return f"missing_khan_id_{doc_idx}_{suffix}"


def _stable_tiny_doc_id(doc: Dict, doc_idx: int) -> str:
    raw = str(doc.get("id") or "").strip()
    if raw and raw.lower() != "unknown":
        return raw
    text = str(doc.get("text") or "").strip()
    seed = text[:500] if text else ""
    suffix = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12] if seed else "na"
    return f"missing_tiny_id_{doc_idx}_{suffix}"


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

def iter_documents():
    """
    Yield (doc_id, text) for every chunk from both datasets.
    doc_id format:
      - "khan::<stable_doc_id>::<chunk_id>"
      - "tiny::<batch>::<doc_id>::<chunk_id>"
    """
    # Khan Academy
    khan_path = Path(KHAN_DATA_PATH)
    if khan_path.exists():
        print("Loading Khan Academy documents...")
        with open(khan_path, "r", encoding="utf-8") as f:
            khan_data = json.load(f)
        seen_khan_ids: Dict[str, int] = {}
        for doc_idx, doc in enumerate(tqdm(khan_data, desc="Khan chunks")):
            text = doc.get("content", "")
            if len(text.strip()) < 50:
                continue
            base_id = _dedupe_with_counter(
                _stable_khan_doc_base_id(doc, doc_idx),
                seen_khan_ids,
            )
            for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                yield f"khan::{base_id}::{i}", chunk

    # Tiny-Textbooks (all batches)
    tiny_dir = Path(TINY_DATA_DIR)
    if tiny_dir.exists():
        batch_files = sorted(tiny_dir.glob("batch_*.json"))
        if INDEX_MAX_BATCHES:
            batch_files = batch_files[:INDEX_MAX_BATCHES]
            print(f"  (index mode: first {INDEX_MAX_BATCHES} batches only)")
        print(f"\nLoading Tiny-Textbooks ({len(batch_files)} batches)...")
        for batch_file in tqdm(batch_files, desc="Tiny batches"):
            with open(batch_file, "r", encoding="utf-8", errors="replace") as f:
                batch = json.load(f)
            seen_tiny_ids: Dict[str, int] = {}
            for doc_idx, doc in enumerate(batch):
                text   = doc.get("text", "")
                if len(text.strip()) < 100:
                    continue
                doc_id = _dedupe_with_counter(
                    _stable_tiny_doc_id(doc, doc_idx),
                    seen_tiny_ids,
                )
                for i, chunk in enumerate(chunk_text(text, CHUNK_SIZE)):
                    yield f"tiny::{batch_file.name}::{doc_id}::{i}", chunk


# ==============================================================================
# Index building
# ==============================================================================

def build_corpus_index():
    Path("outputs").mkdir(exist_ok=True)

    print("="*60)
    print("BUILDING CORPUS INDEX")
    print("="*60)

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
    for doc_id, text in iter_documents():
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
