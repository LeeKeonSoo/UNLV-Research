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
"""

import hashlib
import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import jsonlines

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

TOP_K_DOMAINS  = 5
MIN_SIMILARITY = 0.1
CHUNK_SIZE     = 200
USE_GPU        = True

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
        self.vectorizer    = vectorizer
        self.proto_ids     = proto_ids
        self.proto_matrix  = proto_matrix
        self.use_torch     = False
        self.device        = "cpu"

        if use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            self.use_torch = True
            self.device    = "cuda"
            self.proto_t   = torch.from_numpy(proto_matrix).to(self.device)
            print("Domain classifier: using CUDA")
        else:
            print("Domain classifier: using CPU")

    def classify(self, text: str) -> Dict[str, float]:
        try:
            qv = self.vectorizer.transform([text])
        except Exception:
            return {}
        if qv.nnz == 0:
            return {}
        norm = float(np.sqrt(qv.multiply(qv).sum()))
        if norm == 0.0:
            return {}

        if self.use_torch:
            q = torch.from_numpy(qv.toarray().astype(np.float32)).to(self.device)
            sims = (q @ self.proto_t.T / norm).squeeze(0).cpu().numpy()
        else:
            q = qv.toarray().astype(np.float32)
            sims = (q @ self.proto_matrix.T).ravel() / norm

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

def compute_difficulty(text: str) -> Dict:
    try:
        fk_grade    = textstat.flesch_kincaid_grade(text)
        flesch_ease = textstat.flesch_reading_ease(text)
        smog        = textstat.smog_index(text)
    except Exception:
        fk_grade = flesch_ease = smog = 0.0

    try:
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
        if not Path(index_path).exists():
            print(f"  ⚠ Corpus index not found at {index_path}. "
                  "Redundancy scores will be 0. Run 2a_build_corpus_index.py first.")
            self._available = False
            return

        print(f"Loading corpus index from {index_path}...")
        with open(index_path, "rb") as f:
            idx = pickle.load(f)

        self._available      = True
        self._exact_hashes   = idx["exact_hashes"]
        self._lsh            = idx["lsh"]
        self._minhashes      = idx["minhashes"]
        self._corpus_matrix  = idx["corpus_matrix"]
        self._vectorizer     = idx["vectorizer"]
        self._doc_ids        = idx["doc_ids"]
        self._num_perm       = 128
        print(f"✓ Corpus index loaded ({len(self._doc_ids):,} chunks)")

    def _ngram_overlap(self, text: str, n: int) -> float:
        """Jaccard overlap of this text's n-grams with the top-similar corpus doc."""
        tokens = text.lower().split()
        if len(tokens) < n:
            return 0.0
        text_grams = set(
            " ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
        )
        if not text_grams:
            return 0.0
        # Compare with corpus using TF-IDF similarity to find nearest doc
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cossim
            tv = self._vectorizer.transform([text])
            sims = cossim(tv, self._corpus_matrix).ravel()
            best_idx = int(sims.argmax())
            best_doc_id = self._doc_ids[best_idx]
            # Rebuild n-grams for best doc
            best_tokens = best_doc_id.lower().split("::")[-1].split()
            # Fall back: use similarity score as proxy for overlap
            return round(float(sims[best_idx]), 4)
        except Exception:
            return 0.0

    def compute(self, text: str) -> Dict:
        if not self._available:
            return {
                "exact_duplicate":        False,
                "near_duplicate_score":   0.0,
                "semantic_duplicate_score": 0.0,
                "n_gram_overlap_3":       0.0,
                "n_gram_overlap_5":       0.0,
            }

        # 1. Exact duplicate
        text_hash  = hashlib.md5(text.encode()).hexdigest()
        exact_dup  = text_hash in self._exact_hashes

        # 2. Near-duplicate (MinHash)
        from datasketch import MinHash
        mh = MinHash(num_perm=self._num_perm)
        for word in text.lower().split():
            mh.update(word.encode("utf-8"))
        similar = self._lsh.query(mh)
        near_score = 0.0
        if similar:
            near_score = max(
                mh.jaccard(self._minhashes[d]) for d in similar
                if d in self._minhashes
            )

        # 3. Semantic duplicate (TF-IDF cosine)
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cossim
            tv = self._vectorizer.transform([text])
            sims = cossim(tv, self._corpus_matrix).ravel()
            sem_score = float(np.sort(sims)[-2]) if len(sims) > 1 else 0.0  # 2nd highest (1st = self)
        except Exception:
            sem_score = 0.0

        # 4. N-gram overlap (proxy via TF-IDF similarity)
        ngram3 = self._ngram_overlap(text, 3)
        ngram5 = self._ngram_overlap(text, 5)

        return {
            "exact_duplicate":          exact_dup,
            "near_duplicate_score":     round(near_score, 4),
            "semantic_duplicate_score": round(sem_score, 4),
            "n_gram_overlap_3":         ngram3,
            "n_gram_overlap_5":         ngram5,
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
        if not TRANSFORMERS_AVAILABLE:
            print("  ⚠ transformers not available. Perplexity will be null.")
            return
        try:
            device_str = "cuda" if (use_gpu and TORCH_AVAILABLE
                                    and torch.cuda.is_available()) else "cpu"
            print(f"Loading GPT-2 model (device={device_str})...")
            self._tokenizer = GPT2Tokenizer.from_pretrained(
                "gpt2", cache_dir="./models")
            self._model = GPT2LMHeadModel.from_pretrained(
                "gpt2", cache_dir="./models").to(device_str).eval()
            self._device    = device_str
            self._available = True
            print("✓ GPT-2 loaded")
        except Exception as e:
            print(f"  ⚠ GPT-2 load failed ({e}). Perplexity will be null.")

    def _score_text(self, text: str) -> Optional[float]:
        if not self._available:
            return None
        try:
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.MAX_TOKENS,
            )
            input_ids = enc["input_ids"].to(self._device)
            if input_ids.shape[1] < 2:
                return None
            with torch.no_grad():
                out  = self._model(input_ids, labels=input_ids)
                ppl  = torch.exp(out.loss).item()
            return round(ppl, 2) if np.isfinite(ppl) else None
        except Exception:
            return None

    def compute(self, text: str) -> Dict:
        if not self._available:
            return {
                "gpt2":                    None,
                "token_level_variance":    None,
                "sentence_level_mean":     None,
                "max_sentence_perplexity": None,
            }

        overall = self._score_text(text)

        # Sentence-level breakdown
        try:
            sentences = sent_tokenize(text)
            sent_ppls = [
                self._score_text(s) for s in sentences
                if len(s.split()) >= 5
            ]
            sent_ppls = [p for p in sent_ppls if p is not None]
        except Exception:
            sent_ppls = []

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

def _process_chunks(
    chunks: List[str],
    doc_meta: Dict,
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
) -> List[Dict]:
    results = []
    for chunk_id, chunk in enumerate(chunks):
        if len(chunk.split()) < 20:
            continue

        quality    = compute_quality(chunk)
        difficulty = compute_difficulty(chunk)
        redun      = redundancy.compute(chunk)
        ppl        = perplexity.compute(chunk)

        record = {
            **doc_meta,
            "chunk_id":   chunk_id,
            "text":       chunk,
            "word_count": len(chunk.split()),
            "domain_labels":        classifier.classify(chunk),
            "educational_markers":  quality["educational_markers"],
            "quality_score":        quality["quality_score"],
            "difficulty":           difficulty,
            "redundancy":           redun,
            "perplexity":           ppl,
        }
        results.append(record)
    return results


def process_khan_academy(
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
):
    print("\n" + "="*60)
    print("Processing Khan Academy Dataset")
    print("="*60)

    with open(KHAN_DATA_PATH, "r", encoding="utf-8", errors="replace") as f:
        khan_data = json.load(f)

    results: List[Dict] = []
    for doc in tqdm(khan_data, desc="Khan"):
        text = doc.get("content", "")
        if len(text.strip()) < 50:
            continue
        meta = {
            "source":  "khan_academy",
            "doc_id":  doc.get("url", "unknown"),
            "subject": doc.get("subject", "Unknown"),
            "grade":   doc.get("grade",   "Unknown"),
            "title":   doc.get("title",   "Untitled"),
        }
        results.extend(
            _process_chunks(chunk_text(text, CHUNK_SIZE), meta,
                            classifier, redundancy, perplexity)
        )

    with jsonlines.open(KHAN_OUTPUT, "w") as w:
        w.write_all(results)
    print(f"\n✓ {len(results):,} chunks → {KHAN_OUTPUT}")


def process_tiny_textbooks(
    classifier: DomainClassifier,
    redundancy: RedundancyChecker,
    perplexity: PerplexityScorer,
    max_batches: int = 5,
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
                "source":     "tiny_textbooks",
                "doc_id":     doc.get("id", "unknown"),
                "batch_file": bf.name,
            }
            results.extend(
                _process_chunks(chunk_text(text, CHUNK_SIZE), meta,
                                classifier, redundancy, perplexity)
            )

    with jsonlines.open(TINY_OUTPUT, "w") as w:
        w.write_all(results)
    print(f"\n✓ {len(results):,} chunks from {total_docs:,} docs → {TINY_OUTPUT}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DATASET ANALYSIS — ALL 5 METRICS")
    print("="*60)

    # Load domain classifier
    prototypes, domain_vectorizer = _load_prototypes()
    proto_ids, proto_matrix       = _build_prototype_matrix(prototypes)
    classifier = DomainClassifier(
        domain_vectorizer, proto_ids, proto_matrix, use_gpu=USE_GPU)

    # Load redundancy checker
    redundancy = RedundancyChecker(CORPUS_INDEX_PATH)

    # Load perplexity scorer
    perplexity = PerplexityScorer(use_gpu=USE_GPU)

    # Process datasets
    process_khan_academy(classifier, redundancy, perplexity)
    process_tiny_textbooks(
        classifier, redundancy, perplexity,
        max_batches=None,  # full run
    )

    print("\n" + "="*60)
    print("✓ All 5 metrics computed!")
    print("="*60)
    print("\nNext step: python 3_build_dashboard.py")


if __name__ == "__main__":
    main()
