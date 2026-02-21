"""
Step 2: Compute All 5 Metrics per Paragraph

Metrics computed per chunk:
  1. Domain Coverage  - TF-IDF cosine similarity to Khan course prototypes
  2. Quality          - Educational marker detection + aggregate score
  3. Difficulty       - Flesch-Kincaid, SMOG, lexical diversity, etc.
  4. Redundancy       - Exact hash, MinHash near-dup, n-gram overlap, TF-IDF semantic
  5. Perplexity       - GPT-2 log-loss based (optional; skipped if model unavailable)

Input:
  - outputs/concept_prototypes_tfidf.pkl   (from Step 1)
  - outputs/corpus_index.pkl               (from Step 2a)
  - khan_k12_concepts/all_k12_concepts.json
  - tiny_textbooks_raw/*.json

Output:
  - outputs/khan_analysis.jsonl
  - outputs/tiny_textbooks_analysis.jsonl
  - outputs/run_manifest.json
"""

import hashlib
import json
import os
import pickle
import re
import subprocess
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import jsonlines

# Ensure required NLTK data is present (silent if already downloaded)
for _nltk_pkg in ("punkt_tab", "punkt", "words"):
    nltk.download(_nltk_pkg, quiet=True)

# Suppress numpy matmul warnings (divide-by-zero on zero-norm chunks)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- optional torch ----------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---------- optional transformers (perplexity) ----------
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------- optional datasketch (redundancy) ----------
try:
    from datasketch import MinHash as _MinHash
    REDUNDANCY_DEPS_AVAILABLE = True
except Exception:
    REDUNDANCY_DEPS_AVAILABLE = False


# ==============================================================================
# Configuration
# ==============================================================================

PROTOTYPES_PATH    = "outputs/concept_prototypes_tfidf.pkl"
CORPUS_INDEX_PATH  = "outputs/corpus_index.pkl"
KHAN_DATA_PATH     = "khan_k12_concepts/all_k12_concepts.json"
TINY_TEXTBOOKS_DIR = "tiny_textbooks_raw"

OUTPUT_DIR   = Path("outputs")
KHAN_OUTPUT  = OUTPUT_DIR / "khan_analysis.jsonl"
TINY_OUTPUT  = OUTPUT_DIR / "tiny_textbooks_analysis.jsonl"
RUN_MANIFEST_OUTPUT = OUTPUT_DIR / "run_manifest.json"

TOP_K_DOMAINS  = 5
MIN_SIMILARITY = 0.1
CHUNK_SIZE     = 200
MIN_CHUNK_WORDS = 20
USE_GPU        = True
# Device config (override via env):
#   PHASE1_DEVICE=auto|cuda|mps|cpu
#   PHASE1_CUDA_DEVICE=0
DEVICE_PREFERENCE = os.getenv("PHASE1_DEVICE", "auto").strip().lower()
CUDA_DEVICE       = int(os.getenv("PHASE1_CUDA_DEVICE", "0"))
DOMAIN_BATCH_SIZE = int(os.getenv("PHASE1_DOMAIN_BATCH_SIZE", "256"))
_max_batches_env = os.getenv("PHASE1_MAX_BATCHES", "").strip()
TINY_MAX_BATCHES = int(_max_batches_env) if _max_batches_env else None

SCHEMA_VERSION = "v2"
METRIC_TIER = {
    "domain": "core",
    "quality": "core",
    "difficulty": "core",
    "redundancy": "exploratory",
    "perplexity": "exploratory",
}

DOMAIN_TOP1_GATE = 0.60
DOMAIN_TOP3_GATE = 0.85
QUALITY_PREC_GATE = 0.80
DIFFICULTY_OOR_GATE = 0.01
PERPLEXITY_COVERAGE_GATE = 0.90

# Top-3000 common English words (lightweight proxy for rare-word detection)
_COMMON_WORDS: Optional[set] = None

def _load_common_words() -> set:
    global _COMMON_WORDS
    if _COMMON_WORDS is None:
        # Use NLTK corpus words as a proxy; fallback to a small hardcoded set
        try:
            from nltk.corpus import words as nltk_words
            import nltk
            nltk.download("words", quiet=True)
            _COMMON_WORDS = set(w.lower() for w in nltk_words.words()[:10000])
        except Exception:
            # Absolute fallback: empty set (all words treated as rare → skip rare_words_pct)
            _COMMON_WORDS = set()
    return _COMMON_WORDS


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _hash_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fingerprint_files(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        if not p.exists():
            continue
        st = p.stat()
        row = f"{p.name}|{st.st_size}|{int(st.st_mtime)}\n"
        h.update(row.encode("utf-8"))
    return h.hexdigest()


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _mps_available() -> bool:
    if not TORCH_AVAILABLE:
        return False
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available())


def _resolve_torch_device(
    use_gpu: bool = True,
    preference: str = "auto",
    cuda_device: int = 0,
) -> Tuple[str, str]:
    """
    Returns (device, reason) where device is one of:
      - "cuda:<id>"
      - "mps"
      - "cpu"
    """
    pref = (preference or "auto").lower()
    if not use_gpu:
        return "cpu", "GPU disabled by config"
    if not TORCH_AVAILABLE:
        return "cpu", "torch not available"

    if pref.startswith("cuda"):
        requested = f"cuda:{cuda_device}" if pref == "cuda" else pref
        if torch.cuda.is_available():
            return requested, "forced CUDA"
        return "cpu", f"{requested} requested but CUDA unavailable"

    if pref == "mps":
        if _mps_available():
            return "mps", "forced MPS"
        return "cpu", "MPS requested but unavailable"

    if pref == "cpu":
        return "cpu", "forced CPU"

    # auto
    if torch.cuda.is_available():
        return f"cuda:{cuda_device}", "auto-selected CUDA"
    if _mps_available():
        return "mps", "auto-selected MPS"
    return "cpu", "auto-selected CPU"


def _is_domain_valid(domain_labels: Dict[str, float]) -> bool:
    if not isinstance(domain_labels, dict) or not domain_labels:
        return False
    vals = [v for v in (_safe_float(x) for x in domain_labels.values()) if v is not None]
    if not vals:
        return False
    total = sum(vals)
    return 0.80 <= total <= 1.20


def _is_quality_valid(quality_score: float, markers: Dict) -> bool:
    qs = _safe_float(quality_score)
    if qs is None or qs < 0.0 or qs > 1.0:
        return False
    needed = ("has_examples", "has_explanation", "has_structure")
    if not isinstance(markers, dict):
        return False
    return all(isinstance(markers.get(k), bool) for k in needed)


def _is_difficulty_valid(difficulty: Dict) -> bool:
    if not isinstance(difficulty, dict):
        return False
    fk_grade = _safe_float(difficulty.get("flesch_kincaid_grade"))
    fk_ease = _safe_float(difficulty.get("flesch_reading_ease"))
    smog = _safe_float(difficulty.get("smog_index"))
    ttr = _safe_float(difficulty.get("lexical_diversity"))
    rare = _safe_float(difficulty.get("rare_words_pct"))
    if fk_grade is None or fk_ease is None or smog is None or ttr is None or rare is None:
        return False
    return (
        0.0 <= fk_grade <= 20.0
        and -50.0 <= fk_ease <= 120.0
        and 0.0 <= smog <= 30.0
        and 0.0 <= ttr <= 1.0
        and 0.0 <= rare <= 1.0
    )


def _is_redundancy_valid(redundancy: Dict, available: bool) -> bool:
    if not available or not isinstance(redundancy, dict):
        return False
    exact = redundancy.get("exact_duplicate")
    near = _safe_float(redundancy.get("near_duplicate_score"))
    sem = _safe_float(redundancy.get("semantic_duplicate_score"))
    ng3 = _safe_float(redundancy.get("n_gram_overlap_3"))
    ng5 = _safe_float(redundancy.get("n_gram_overlap_5"))
    if not isinstance(exact, bool):
        return False
    numeric_ok = all(v is not None and 0.0 <= v <= 1.0 for v in (near, sem, ng3, ng5))
    return numeric_ok


def _is_perplexity_valid(perplexity: Dict) -> bool:
    if not isinstance(perplexity, dict):
        return False
    return _safe_float(perplexity.get("gpt2")) is not None


class RunningStat:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min = None
        self.max = None

    def add(self, value: Optional[float]):
        if value is None:
            return
        x = float(value)
        self.n += 1
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return float(np.sqrt(self.m2 / (self.n - 1)))

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "count": self.n,
            "mean": round(self.mean, 6) if self.n else None,
            "std": round(self.std, 6) if self.n else None,
            "min": round(self.min, 6) if self.min is not None else None,
            "max": round(self.max, 6) if self.max is not None else None,
        }


class ReliabilityTracker:
    def __init__(self):
        self.total_chunks = 0
        self.perplexity_non_null = 0
        self.difficulty_out_of_range = 0
        self.exact_dup_true = 0
        self.near_stats = RunningStat()
        self.semantic_stats = RunningStat()
        self.ngram3_stats = RunningStat()
        self.ngram5_stats = RunningStat()

    def observe(self, record: Dict):
        self.total_chunks += 1

        d = record.get("difficulty") or {}
        if not _is_difficulty_valid(d):
            self.difficulty_out_of_range += 1

        r = record.get("redundancy") or {}
        self.exact_dup_true += int(bool(r.get("exact_duplicate")))
        self.near_stats.add(_safe_float(r.get("near_duplicate_score")))
        self.semantic_stats.add(_safe_float(r.get("semantic_duplicate_score")))
        self.ngram3_stats.add(_safe_float(r.get("n_gram_overlap_3")))
        self.ngram5_stats.add(_safe_float(r.get("n_gram_overlap_5")))

        p = record.get("perplexity") or {}
        if _safe_float(p.get("gpt2")) is not None:
            self.perplexity_non_null += 1

    def gate_outcomes(self) -> Dict:
        n = max(self.total_chunks, 1)
        difficulty_oor_rate = self.difficulty_out_of_range / n
        perplexity_coverage = self.perplexity_non_null / n
        exact_dup_rate = self.exact_dup_true / n
        near_std = self.near_stats.std
        semantic_std = self.semantic_stats.std
        ngram_std = max(self.ngram3_stats.std, self.ngram5_stats.std)
        non_degenerate = (
            (near_std > 1e-6 or semantic_std > 1e-6 or ngram_std > 1e-6)
            and exact_dup_rate < 0.99
        )

        return {
            "domain": {
                "top1_accuracy": {
                    "threshold": DOMAIN_TOP1_GATE,
                    "value": None,
                    "pass": None,
                    "status": "requires_manual_validation_set",
                },
                "top3_recall": {
                    "threshold": DOMAIN_TOP3_GATE,
                    "value": None,
                    "pass": None,
                    "status": "requires_manual_validation_set",
                },
            },
            "quality": {
                "macro_precision": {
                    "threshold": QUALITY_PREC_GATE,
                    "value": None,
                    "pass": None,
                    "status": "requires_manual_validation_set",
                }
            },
            "difficulty": {
                "out_of_range_rate": {
                    "threshold": DIFFICULTY_OOR_GATE,
                    "value": round(difficulty_oor_rate, 6),
                    "pass": difficulty_oor_rate <= DIFFICULTY_OOR_GATE,
                    "status": "auto_computed",
                }
            },
            "redundancy": {
                "non_degenerate_distribution": {
                    "threshold": "std>0 and exact_dup_rate<0.99",
                    "value": {
                        "exact_duplicate_rate": round(exact_dup_rate, 6),
                        "near_duplicate_std": round(near_std, 6),
                        "semantic_duplicate_std": round(semantic_std, 6),
                        "ngram_overlap_std": round(ngram_std, 6),
                    },
                    "pass": non_degenerate,
                    "status": "auto_computed",
                }
            },
            "perplexity": {
                "non_null_coverage": {
                    "threshold": PERPLEXITY_COVERAGE_GATE,
                    "value": round(perplexity_coverage, 6),
                    "pass": perplexity_coverage >= PERPLEXITY_COVERAGE_GATE,
                    "status": "auto_computed",
                }
            },
            "distribution_summary": {
                "near_duplicate": self.near_stats.as_dict(),
                "semantic_duplicate": self.semantic_stats.as_dict(),
                "n_gram_overlap_3": self.ngram3_stats.as_dict(),
                "n_gram_overlap_5": self.ngram5_stats.as_dict(),
            },
        }

# ==============================================================================
# 1. Domain Classification
# ==============================================================================

def _load_prototypes():
    with open(PROTOTYPES_PATH, "rb") as f:
        data = pickle.load(f)
    return data["prototypes"], data["vectorizer"]


def _build_prototype_matrix(prototypes: Dict) -> Tuple[List[str], np.ndarray]:
    ids = list(prototypes.keys())
    matrix = np.vstack([prototypes[i] for i in ids]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, matrix / norms


class DomainClassifier:
    def __init__(self, vectorizer, proto_ids, proto_matrix, use_gpu=True):
        self.vectorizer = vectorizer
        self.proto_ids = proto_ids
        self.proto_matrix = proto_matrix
        self.use_torch = False
        self.device, reason = _resolve_torch_device(
            use_gpu=use_gpu,
            preference=DEVICE_PREFERENCE,
            cuda_device=CUDA_DEVICE,
        )

        if TORCH_AVAILABLE and self.device.startswith("cuda"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass

        if TORCH_AVAILABLE and self.device != "cpu":
            self.use_torch = True
            self.proto_t = torch.from_numpy(proto_matrix).to(self.device)
            print(f"Domain classifier: using {self.device} ({reason})")
        else:
            print(f"Domain classifier: using CPU ({reason})")

    def _to_label_distribution(self, sims: np.ndarray) -> Dict[str, float]:
        valid = np.where(sims >= MIN_SIMILARITY)[0]
        if valid.size == 0:
            return {}
        if valid.size > TOP_K_DOMAINS:
            top = valid[np.argpartition(sims[valid], -TOP_K_DOMAINS)[-TOP_K_DOMAINS:]]
        else:
            top = valid
        top = top[np.argsort(sims[top])[::-1]]
        result = {self.proto_ids[i]: float(sims[i]) for i in top}
        total = sum(result.values())
        return {k: v / total for k, v in result.items()} if total > 0 else result

    def classify_batch(
        self,
        texts: List[str],
        batch_size: int = DOMAIN_BATCH_SIZE,
    ) -> List[Dict[str, float]]:
        if not texts:
            return []

        outputs: List[Dict[str, float]] = []
        for i in range(0, len(texts), max(batch_size, 1)):
            b_texts = texts[i : i + max(batch_size, 1)]
            try:
                qv = self.vectorizer.transform(b_texts)
            except Exception:
                outputs.extend({} for _ in b_texts)
                continue

            q = qv.toarray().astype(np.float32)
            norms = np.linalg.norm(q, axis=1)

            if self.use_torch:
                qt = torch.from_numpy(q).to(self.device)
                with torch.inference_mode():
                    sims_mat = (qt @ self.proto_t.T).cpu().numpy()
            else:
                sims_mat = q @ self.proto_matrix.T

            for row_idx in range(len(b_texts)):
                norm = float(norms[row_idx])
                if norm <= 0.0:
                    outputs.append({})
                    continue
                sims = sims_mat[row_idx] / norm
                outputs.append(self._to_label_distribution(sims))

        return outputs

    def classify(self, text: str) -> Dict[str, float]:
        return self.classify_batch([text], batch_size=1)[0]


# ==============================================================================
# 2. Quality
# ==============================================================================

_EXAMPLE_MARKERS     = ["for example", "such as", "consider", "let's look at", "instance"]
_EXPLANATION_MARKERS = ["because", "therefore", "this means", "as a result", "consequently"]
_STRUCTURE_MARKERS   = ["first", "second", "third", "finally", "in summary", "in conclusion"]

def compute_quality(text: str) -> Dict:
    t = text.lower()
    has_ex  = any(m in t for m in _EXAMPLE_MARKERS)
    has_exp = any(m in t for m in _EXPLANATION_MARKERS)
    has_str = any(m in t for m in _STRUCTURE_MARKERS)
    score   = (has_ex + has_exp + has_str) / 3.0
    return {
        "educational_markers": {
            "has_examples":    has_ex,
            "has_explanation": has_exp,
            "has_structure":   has_str,
        },
        "quality_score": round(score, 4),
    }


# ==============================================================================
# 3. Difficulty
# ==============================================================================

def compute_difficulty(text: str, sentences: List[str] = None) -> Dict:
    try:
        fk_grade    = textstat.flesch_kincaid_grade(text)
        flesch_ease = textstat.flesch_reading_ease(text)
        smog        = textstat.smog_index(text)
    except Exception:
        fk_grade = flesch_ease = smog = 0.0

    try:
        if sentences is None:
            sentences = sent_tokenize(text)
        words     = [w for w in word_tokenize(text) if w.isalpha()]
        n_sents   = max(len(sentences), 1)
        n_words   = max(len(words), 1)

        avg_sent_len  = round(n_words / n_sents, 2)
        avg_word_len  = round(sum(len(w) for w in words) / n_words, 2)
        ttr           = round(len(set(w.lower() for w in words)) / n_words, 4)

        common = _load_common_words()
        if common:
            rare_count = sum(1 for w in words if w.lower() not in common)
            rare_pct   = round(rare_count / n_words, 4)
        else:
            rare_pct = 0.0
    except Exception:
        avg_sent_len = avg_word_len = ttr = rare_pct = 0.0

    return {
        "flesch_kincaid_grade": round(fk_grade, 2),
        "flesch_reading_ease":  round(flesch_ease, 2),
        "smog_index":           round(smog, 2),
        "avg_sentence_length":  avg_sent_len,
        "avg_word_length":      avg_word_len,
        "rare_words_pct":       rare_pct,
        "lexical_diversity":    ttr,
    }


# ==============================================================================
# 4. Redundancy
# ==============================================================================

class RedundancyChecker:
    """
    Wraps the pre-built corpus index to compute per-chunk redundancy metrics.
    Falls back to zeros when corpus index is unavailable.
    """

    def __init__(self, index_path: str):
        if not REDUNDANCY_DEPS_AVAILABLE:
            print("  ⚠ datasketch not available. Redundancy scores will be 0.")
            self._available = False
            return

        if not Path(index_path).exists():
            print(
                f"  ⚠ Corpus index not found at {index_path}. "
                "Redundancy scores will be 0. Run build_corpus_index.py first."
            )
            self._available = False
            return

        print(f"Loading corpus index from {index_path}...")
        with open(index_path, "rb") as f:
            idx = pickle.load(f)

        self._available = True
        self._exact_hash_counts = idx.get("exact_hash_counts") or {
            h: 1 for h in idx.get("exact_hashes", set())
        }
        self._doc_hash_by_id = idx.get("doc_hash_by_id", {})
        self._doc_texts = idx.get("doc_texts", {})
        self._lsh = idx["lsh"]
        self._minhashes = idx["minhashes"]
        self._num_perm = 128
        self._MinHash = _MinHash
        self._vectorizer = idx.get("vectorizer")
        self._corpus_matrix = idx.get("corpus_matrix")
        self._doc_ids = idx.get("doc_ids", [])
        self._doc_row_index = {doc_id: i for i, doc_id in enumerate(self._doc_ids)}
        self._doc_ids_by_hash: Dict[str, List[str]] = {}
        self._query_vec_cache: Dict[str, object] = {}
        self._ngram_cache_3: Dict[str, set] = {}
        self._ngram_cache_5: Dict[str, set] = {}
        self._cache_limit = 20000
        for doc_id, h in self._doc_hash_by_id.items():
            self._doc_ids_by_hash.setdefault(h, []).append(doc_id)
        print(f"✓ Corpus index loaded ({len(self._minhashes):,} chunks)")

    @property
    def available(self) -> bool:
        return self._available

    @staticmethod
    def _token_ngrams(text: str, n: int) -> set:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < n:
            return set()
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        return inter / union if union else 0.0

    def _cache_put(self, cache: Dict, key, value):
        if len(cache) >= self._cache_limit:
            cache.clear()
        cache[key] = value

    def _vectorize_query(self, text: str):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        cached = self._query_vec_cache.get(key)
        if cached is not None:
            return cached
        qv = self._vectorizer.transform([text])
        self._cache_put(self._query_vec_cache, key, qv)
        return qv

    def _doc_ngrams(self, doc_id: str, text: str, n: int) -> set:
        cache = self._ngram_cache_3 if n == 3 else self._ngram_cache_5
        cached = cache.get(doc_id)
        if cached is not None:
            return cached
        grams = self._token_ngrams(text, n)
        self._cache_put(cache, doc_id, grams)
        return grams

    def _semantic_score(self, text: str, candidate_ids: List[str]) -> float:
        if (
            not candidate_ids
            or self._vectorizer is None
            or self._corpus_matrix is None
            or not self._doc_row_index
        ):
            return 0.0
        row_ids = [self._doc_row_index[c] for c in candidate_ids if c in self._doc_row_index]
        if not row_ids:
            return 0.0
        try:
            qv = self._vectorize_query(text)
            cand_matrix = self._corpus_matrix[row_ids]
            sims = qv @ cand_matrix.T
            if sims.nnz == 0:
                return 0.0
            return float(sims.data.max())
        except Exception:
            return 0.0

    def _ngram_scores(self, text: str, candidate_ids: List[str]) -> Tuple[float, float]:
        if not candidate_ids or not self._doc_texts:
            return 0.0, 0.0
        q3 = self._token_ngrams(text, 3)
        q5 = self._token_ngrams(text, 5)
        best3, best5 = 0.0, 0.0
        for cid in candidate_ids[:200]:
            cand_text = self._doc_texts.get(cid)
            if not cand_text:
                continue
            c3 = self._doc_ngrams(cid, cand_text, 3)
            c5 = self._doc_ngrams(cid, cand_text, 5)
            best3 = max(best3, self._jaccard(q3, c3))
            best5 = max(best5, self._jaccard(q5, c5))
        return best3, best5

    def compute(self, text: str, current_doc_id: Optional[str] = None) -> Dict:
        if not self._available:
            return {
                "exact_duplicate": False,
                "near_duplicate_score": 0.0,
                "semantic_duplicate_score": 0.0,
                "n_gram_overlap_3": 0.0,
                "n_gram_overlap_5": 0.0,
            }

        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        hash_count = int(self._exact_hash_counts.get(text_hash, 0))
        exact_dup = hash_count > 1
        hash_bucket_ids = list(self._doc_ids_by_hash.get(text_hash, []))
        hash_bucket_set = set(hash_bucket_ids)

        mh = self._MinHash(num_perm=self._num_perm)
        for word in set(text.lower().split()):
            mh.update(word.encode("utf-8"))

        candidate_ids = set(self._lsh.query(mh))
        if hash_bucket_ids:
            candidate_ids.update(hash_bucket_ids)
        if current_doc_id:
            candidate_ids.discard(current_doc_id)

        # Robust self-match mitigation:
        # If current_doc_id is missing/misaligned, remove one deterministic hash-bucket
        # candidate so a chunk is less likely to match itself at score~1 by default.
        if hash_bucket_ids and (not current_doc_id or current_doc_id not in hash_bucket_set):
            candidate_ids.discard(min(hash_bucket_ids))

        near_score = 0.0
        if candidate_ids:
            scores = [
                mh.jaccard(self._minhashes[cid])
                for cid in candidate_ids
                if cid in self._minhashes
            ]
            if scores:
                near_score = float(max(scores))

        hash_first = [cid for cid in sorted(hash_bucket_ids) if cid in candidate_ids]
        others = [cid for cid in candidate_ids if cid not in set(hash_first)]
        cand_list = hash_first + others
        semantic_score = self._semantic_score(text, cand_list)
        ngram3, ngram5 = self._ngram_scores(text, cand_list)

        return {
            "exact_duplicate": bool(exact_dup),
            "near_duplicate_score": round(near_score, 4),
            "semantic_duplicate_score": round(semantic_score, 4),
            "n_gram_overlap_3": round(ngram3, 4),
            "n_gram_overlap_5": round(ngram5, 4),
        }


# ==============================================================================
# 5. Perplexity (GPT-2)
# ==============================================================================

class PerplexityScorer:
    """
    Computes GPT-2 perplexity.  Gracefully returns None values if the model
    is unavailable or text is too long.
    """
    MAX_TOKENS = 512

    def __init__(self, use_gpu: bool = True):
        self._available = False
        self._device = "cpu"
        self._batch_error_logged = False
        if not TRANSFORMERS_AVAILABLE:
            print("  ⚠ transformers not available. Perplexity will be null.")
            return
        try:
            device_str, reason = _resolve_torch_device(
                use_gpu=use_gpu,
                preference=DEVICE_PREFERENCE,
                cuda_device=CUDA_DEVICE,
            )
            self._device = device_str
            print(f"Loading GPT-2 model (device={device_str}, {reason})...")

            self._tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2", cache_dir="./models")
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            model_kwargs = {"cache_dir": "./models"}
            if device_str.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16

            self._model = GPT2LMHeadModel.from_pretrained(
                "gpt2", **model_kwargs
            ).to(device_str).eval()
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            self._available = True
            print("✓ GPT-2 loaded")
        except Exception as e:
            print(f"  ⚠ GPT-2 load failed ({e}). Perplexity will be null.")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def device(self) -> str:
        return self._device

    def _score_batch(self, texts: List[str]) -> List[Optional[float]]:
        """Score a list of texts in a single GPU forward pass (with padding)."""
        if not self._available or not texts:
            return [None] * len(texts)
        try:
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_TOKENS,
                padding=True,
            )
            input_ids      = enc["input_ids"].to(self._device)       # (B, T)
            attention_mask = enc["attention_mask"].to(self._device)   # (B, T)
            if input_ids.shape[1] < 2:
                return [None] * len(texts)
            with torch.inference_mode():
                logits = self._model(input_ids).logits                # (B, T, V)
            # Causal LM loss: each token predicts the next
            shift_logits = logits[:, :-1, :].contiguous()             # (B, T-1, V)
            shift_labels = input_ids[:, 1:].contiguous()              # (B, T-1)
            shift_mask   = attention_mask[:, 1:].float().contiguous() # (B, T-1)
            loss_fn      = torch.nn.CrossEntropyLoss(reduction="none")
            token_loss   = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.size(0), -1)                          # (B, T-1)
            # Average over non-padding tokens only
            seq_lens  = shift_mask.sum(dim=1).clamp(min=1)
            mean_loss = (token_loss * shift_mask).sum(dim=1) / seq_lens
            ppls      = torch.exp(mean_loss)
            return [
                round(p.item(), 2) if np.isfinite(p.item()) else None
                for p in ppls
            ]
        except Exception as e:
            if not self._batch_error_logged:
                print(f"  ⚠ GPT-2 scoring failed once ({e}). Returning null perplexity.")
                self._batch_error_logged = True
            return [None] * len(texts)

    def compute(self, text: str, sentences: List[str] = None) -> Dict:
        if not self._available:
            return {
                "gpt2":                    None,
                "token_level_variance":    None,
                "sentence_level_mean":     None,
                "max_sentence_perplexity": None,
            }

        if sentences is None:
            sentences = sent_tokenize(text)

        # Filter sentences by minimum length
        valid_sents = [s for s in sentences if len(s.split()) >= 5]

        # Batch: whole chunk first, then valid sentences — single forward pass
        all_texts = [text] + valid_sents
        all_ppls  = self._score_batch(all_texts)

        overall   = all_ppls[0]
        sent_ppls = [p for p in all_ppls[1:] if p is not None]

        sent_mean = round(float(np.mean(sent_ppls)), 2) if sent_ppls else None
        sent_max  = round(float(np.max(sent_ppls)),  2) if sent_ppls else None
        sent_var  = round(float(np.std(sent_ppls)),  2) if len(sent_ppls) > 1 else 0.0

        return {
            "gpt2":                    overall,
            "token_level_variance":    sent_var,
            "sentence_level_mean":     sent_mean,
            "max_sentence_perplexity": sent_max,
        }


# ==============================================================================
# Text Chunking
# ==============================================================================

def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        words = para.split()
        if len(words) <= chunk_size:
            chunks.append(para)
        else:
            sentences   = re.split(r'(?<=[.!?])\s+', para)
            cur: List[str] = []
            cur_len     = 0
            for sent in sentences:
                sw = len(sent.split())
                if cur_len + sw > chunk_size and cur:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [sent], sw
                else:
                    cur.append(sent)
                    cur_len += sw
            if cur:
                chunks.append(" ".join(cur))
    return chunks


# ==============================================================================
# Dataset Processing
# ==============================================================================

def _build_index_chunk_id(doc_meta: Dict, chunk_id: int) -> Optional[str]:
    source = doc_meta.get("source")
    if source == "khan_academy":
        return f"khan::{doc_meta.get('doc_id', 'unknown')}::{chunk_id}"
    if source == "tiny_textbooks":
        return (
            f"tiny::{doc_meta.get('batch_file', 'unknown')}"
            f"::{doc_meta.get('doc_id', 'unknown')}::{chunk_id}"
        )
    return None


def _build_validity_flags(
    domain_labels: Dict,
    quality_score: float,
    markers: Dict,
    difficulty: Dict,
    redundancy_metrics: Dict,
    redundancy_available: bool,
    perplexity_metrics: Dict,
) -> Dict[str, bool]:
    return {
        "domain_valid": _is_domain_valid(domain_labels),
        "quality_valid": _is_quality_valid(quality_score, markers),
        "difficulty_valid": _is_difficulty_valid(difficulty),
        "redundancy_valid": _is_redundancy_valid(
            redundancy_metrics, redundancy_available
        ),
        "perplexity_valid": _is_perplexity_valid(perplexity_metrics),
    }


def _process_chunks(
    chunks: List[str],
    doc_meta: Dict,
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
    tracker: Optional[ReliabilityTracker] = None,
) -> List[Dict]:
    candidates = []
    for chunk in chunks:
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            continue
        # Match Step 2a index IDs, which enumerate only >=20-word chunks.
        chunk_id = len(candidates)
        sentences = sent_tokenize(chunk)
        candidates.append((chunk_id, chunk, sentences))

    if not candidates:
        return []

    # Batch domain classification improves CUDA/MPS utilization.
    batched_domains = classifier.classify_batch(
        [chunk for _, chunk, _ in candidates],
        batch_size=DOMAIN_BATCH_SIZE,
    )

    results = []
    for (chunk_id, chunk, sentences), domain_labels in zip(candidates, batched_domains):
        quality = compute_quality(chunk)
        difficulty = compute_difficulty(chunk, sentences=sentences)
        index_chunk_id = _build_index_chunk_id(doc_meta, chunk_id)
        redun = redundancy.compute(chunk, current_doc_id=index_chunk_id)
        ppl = perplexity.compute(chunk, sentences=sentences)
        validity_flags = _build_validity_flags(
            domain_labels=domain_labels,
            quality_score=quality["quality_score"],
            markers=quality["educational_markers"],
            difficulty=difficulty,
            redundancy_metrics=redun,
            redundancy_available=redundancy.available,
            perplexity_metrics=ppl,
        )

        record = {
            **doc_meta,
            "schema_version": SCHEMA_VERSION,
            "chunk_id": chunk_id,
            "text": chunk,
            "word_count": len(chunk.split()),
            "domain_labels": domain_labels,
            "educational_markers": quality["educational_markers"],
            "quality_score": quality["quality_score"],
            "difficulty": difficulty,
            "redundancy": redun,
            "perplexity": ppl,
            "metric_tier": dict(METRIC_TIER),
            "validity_flags": validity_flags,
        }
        results.append(record)
        if tracker is not None:
            tracker.observe(record)
    return results


def process_khan_academy(
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
    tracker: Optional[ReliabilityTracker] = None,
):
    print("\n" + "="*60)
    print("Processing Khan Academy Dataset")
    print("="*60)

    with open(KHAN_DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        khan_data = json.load(f)

    results: List[Dict] = []
    docs_used = 0
    for doc in tqdm(khan_data, desc="Khan"):
        text = doc.get("content", "")
        if len(text.strip()) < 50:
            continue
        docs_used += 1
        meta = {
            "source": "khan_academy",
            "doc_id": doc.get("doc_id") or doc.get("url", "unknown"),
            "subject": doc.get("subject", "Unknown"),
            "grade": doc.get("grade", "Unknown"),
            "title": doc.get("title", "Untitled"),
        }
        results.extend(
            _process_chunks(
                chunk_text(text, CHUNK_SIZE),
                meta,
                classifier,
                redundancy,
                perplexity,
                tracker=tracker,
            )
        )

    with jsonlines.open(KHAN_OUTPUT, "w") as w:
        w.write_all(results)
    print(f"\n✓ {len(results):,} chunks → {KHAN_OUTPUT}")
    return {
        "dataset": "khan_academy",
        "input_path": KHAN_DATA_PATH,
        "docs_total": len(khan_data),
        "docs_used": docs_used,
        "chunks_written": len(results),
    }


def process_tiny_textbooks(
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
    max_batches: Optional[int] = 5,
    tracker: Optional[ReliabilityTracker] = None,
):
    print("\n" + "="*60)
    print("Processing Tiny-Textbooks Dataset")
    print("="*60)

    batch_files = sorted(Path(TINY_TEXTBOOKS_DIR).glob("batch_*.json"))
    if max_batches:
        batch_files = batch_files[:max_batches]
        print(f"  (first {max_batches} batches)")

    results: List[Dict] = []
    total_docs = 0
    for bf in tqdm(batch_files, desc="Tiny batches"):
        with open(bf, "r", encoding="utf-8", errors="replace") as f:
            batch = json.load(f)
        for doc in batch:
            total_docs += 1
            text = doc.get("text", "")
            if len(text.strip()) < 100:
                continue
            meta = {
                "source": "tiny_textbooks",
                "doc_id": doc.get("id", "unknown"),
                "batch_file": bf.name,
            }
            results.extend(
                _process_chunks(
                    chunk_text(text, CHUNK_SIZE),
                    meta,
                    classifier,
                    redundancy,
                    perplexity,
                    tracker=tracker,
                )
            )

    with jsonlines.open(TINY_OUTPUT, "w") as w:
        w.write_all(results)
    print(f"\n✓ {len(results):,} chunks from {total_docs:,} docs → {TINY_OUTPUT}")
    return {
        "dataset": "tiny_textbooks",
        "input_dir": TINY_TEXTBOOKS_DIR,
        "batch_count": len(batch_files),
        "batch_files": [bf.name for bf in batch_files],
        "docs_total": total_docs,
        "chunks_written": len(results),
    }


# ==============================================================================
# Main
# ==============================================================================

def build_run_manifest(
    khan_stats: Dict,
    tiny_stats: Dict,
    classifier: DomainClassifier,
    tracker: ReliabilityTracker,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
) -> Dict:
    khan_path = Path(khan_stats.get("input_path", KHAN_DATA_PATH))
    tiny_files = sorted(Path(TINY_TEXTBOOKS_DIR).glob("batch_*.json"))
    if tiny_stats.get("batch_count"):
        tiny_files = tiny_files[: tiny_stats["batch_count"]]

    gate_outcomes = tracker.gate_outcomes()
    perplexity_gate = gate_outcomes["perplexity"]["non_null_coverage"]

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "phase-1",
        "objective_mode": "descriptive_comparative_only",
        "generated_by": "compute_metrics.py",
        "code_commit_hash": _git_commit_hash(),
        "dataset_versions_and_counts": {
            "khan_academy": {
                "input_path": str(khan_path),
                "sha256": _hash_file(khan_path),
                "docs_total": khan_stats.get("docs_total"),
                "docs_used": khan_stats.get("docs_used"),
                "chunks_written": khan_stats.get("chunks_written"),
            },
            "tiny_textbooks": {
                "input_dir": tiny_stats.get("input_dir"),
                "batch_count": tiny_stats.get("batch_count"),
                "batch_manifest_sha256": _fingerprint_files(tiny_files),
                "docs_total": tiny_stats.get("docs_total"),
                "chunks_written": tiny_stats.get("chunks_written"),
            },
        },
        "threshold_config": {
            "TOP_K_DOMAINS": TOP_K_DOMAINS,
            "MIN_SIMILARITY": MIN_SIMILARITY,
            "CHUNK_SIZE": CHUNK_SIZE,
            "DOMAIN_BATCH_SIZE": DOMAIN_BATCH_SIZE,
        },
        "runtime_device": {
            "requested": DEVICE_PREFERENCE,
            "domain_classifier": classifier.device,
            "perplexity_model": perplexity.device,
            "cuda_available": bool(TORCH_AVAILABLE and torch.cuda.is_available()),
            "mps_available": bool(_mps_available()),
        },
        "metric_tier": dict(METRIC_TIER),
        "reliability_gate_outcomes": gate_outcomes,
        "perplexity_fallback_behavior": {
            "model_available": perplexity.available,
            "coverage_on_this_run": perplexity_gate.get("value"),
            "remains_exploratory": not bool(perplexity_gate.get("pass")),
            "policy": (
                "Perplexity remains exploratory unless non-null coverage "
                f">= {PERPLEXITY_COVERAGE_GATE:.2f}."
            ),
        },
        "redundancy_runtime_status": {
            "index_available": redundancy.available,
            "policy": (
                "Redundancy is exploratory and non-claimable until "
                "distribution validity gate passes."
            ),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DATASET ANALYSIS — ALL 5 METRICS")
    print("="*60)
    print(f"Device preference: {DEVICE_PREFERENCE} (CUDA device index: {CUDA_DEVICE})")
    if TINY_MAX_BATCHES is None:
        print("Tiny-Textbooks batch mode: full run")
    else:
        print(f"Tiny-Textbooks batch mode: first {TINY_MAX_BATCHES} batch(es)")

    # Load domain classifier
    prototypes, domain_vectorizer = _load_prototypes()
    proto_ids, proto_matrix       = _build_prototype_matrix(prototypes)
    classifier = DomainClassifier(
        domain_vectorizer, proto_ids, proto_matrix, use_gpu=USE_GPU)

    # Load redundancy checker
    redundancy = RedundancyChecker(CORPUS_INDEX_PATH)

    # Load perplexity scorer
    perplexity = PerplexityScorer(use_gpu=USE_GPU)

    tracker = ReliabilityTracker()

    # Process datasets
    khan_stats = process_khan_academy(
        classifier, redundancy, perplexity, tracker=tracker
    )
    tiny_stats = process_tiny_textbooks(
        classifier,
        redundancy,
        perplexity,
        max_batches=TINY_MAX_BATCHES,
        tracker=tracker,
    )

    manifest = build_run_manifest(
        khan_stats=khan_stats,
        tiny_stats=tiny_stats,
        classifier=classifier,
        tracker=tracker,
        redundancy=redundancy,
        perplexity=perplexity,
    )
    with open(RUN_MANIFEST_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Run manifest saved → {RUN_MANIFEST_OUTPUT}")

    print("\n" + "="*60)
    print("✓ All 5 metrics computed!")
    print("="*60)
    print("\nNext step: python build_dashboard.py")


if __name__ == "__main__":
    main()
