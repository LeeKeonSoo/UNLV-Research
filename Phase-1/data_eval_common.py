#!/usr/bin/env python3
"""Common helpers for the generic data evaluation pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
INDEX_DIR = OUTPUT_DIR / "index"
SCORED_DIR = OUTPUT_DIR / "scored"
SUBSETS_DIR = OUTPUT_DIR / "subsets"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
METRIC_MATURITY_SNAPSHOT_PATH = OUTPUT_DIR / "metric_maturity_snapshot.json"
TOKEN_BUDGET_CACHE_PATH = PROJECT_DIR / "validation" / "token_budget_cache.json"
METRIC_SPEC_PATH = PROJECT_DIR / "configs" / "metric_spec_with_citations.json"
METRIC_MATURITY_TRACKER_CONFIG_PATH = PROJECT_DIR / "configs" / "metric_maturity_tracker.json"
QUALITY_REFERENCE_MODEL_PATH = PROJECT_DIR / "models" / "reference_quality_model.joblib"
QUALITY_REFERENCE_META_PATH = PROJECT_DIR / "models" / "reference_quality_model.meta.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
RUN_SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"
UTILITY_PROBE_RESULTS_PATH = OUTPUT_DIR / "utility_probe_results.json"

SCHEMA_VERSION = "data-eval-v2"
PROFILE_SCHEMA_VERSION = "curation-profiles-v2"
METRIC_SPEC_SCHEMA_VERSION = "metric-spec-v2"
DEFAULT_DATASET_CONFIG = PROJECT_DIR / "datasets_config.json"
DEFAULT_PROFILE_CONFIG = PROJECT_DIR / "configs" / "curation_profiles.json"

CORE_SELECTION_METRICS = (
    "structural_validity_gate",
    "reference_quality_score",
    "exact_duplicate_indicator",
    "shingle_near_duplicate_indicator",
    "shingle_near_duplicate_risk_score",
)
CORE_SUBSET_METRICS = (
    "subset_coverage_retention_score",
    "small_lm_probe_gain_score",
)
DIAGNOSTIC_METRICS = (
    "structural_validity_score",
    "explanatory_quality_proxy",
    "tail_cluster_rarity_proxy",
    "predictive_utility_proxy",
)
SUBSET_DIAGNOSTIC_METRICS = (
    "fixed_token_probe_gain_score",
)
CANONICAL_CORE_METRICS = CORE_SELECTION_METRICS + CORE_SUBSET_METRICS
ALL_METRICS = CANONICAL_CORE_METRICS + DIAGNOSTIC_METRICS + SUBSET_DIAGNOSTIC_METRICS

CHUNK_SIZE = 200
MIN_CHUNK_WORDS = 20

SUPPORTED_FORMATS = {"json_list", "json_batch_dir"}
DEFAULT_TOKENIZER_NAME = "gpt2"
_TOKENIZER_CACHE: Dict[str, Any] = {}
RESILIENT_IO_MAX_RETRIES = int(os.environ.get("DATA_EVAL_RESILIENT_IO_MAX_RETRIES", "600"))
RESILIENT_IO_RETRY_DELAY_SEC = float(os.environ.get("DATA_EVAL_RESILIENT_IO_RETRY_DELAY_SEC", "0.5"))
RESILIENT_IO_MAX_RETRY_DELAY_SEC = float(os.environ.get("DATA_EVAL_RESILIENT_IO_MAX_RETRY_DELAY_SEC", "8.0"))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _is_timeout_error(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or (isinstance(exc, OSError) and getattr(exc, "errno", None) == 60)


def iter_nonempty_lines_resilient(
    path: Path,
    *,
    max_retries: int = RESILIENT_IO_MAX_RETRIES,
    retry_delay_sec: float = RESILIENT_IO_RETRY_DELAY_SEC,
    max_retry_delay_sec: float = RESILIENT_IO_MAX_RETRY_DELAY_SEC,
) -> Iterator[str]:
    offset = 0
    retries = 0
    while True:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                if offset:
                    f.seek(offset)
                while True:
                    offset = f.tell()
                    raw = f.readline()
                    if raw == "":
                        return
                    retries = 0
                    stripped = raw.strip()
                    if stripped:
                        yield stripped
        except OSError as exc:
            if not _is_timeout_error(exc):
                raise
            retries += 1
            if retries > max_retries:
                raise
            delay = min(float(max_retry_delay_sec), float(retry_delay_sec) * (1.35 ** max(retries - 1, 0)))
            time.sleep(delay)


def iter_jsonl_records_resilient(
    path: Path,
    *,
    max_retries: int = RESILIENT_IO_MAX_RETRIES,
    retry_delay_sec: float = RESILIENT_IO_RETRY_DELAY_SEC,
    max_retry_delay_sec: float = RESILIENT_IO_MAX_RETRY_DELAY_SEC,
) -> Iterator[Dict[str, Any]]:
    for raw in iter_nonempty_lines_resilient(
        path,
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        max_retry_delay_sec=max_retry_delay_sec,
    ):
        yield json.loads(raw)


def count_nonempty_lines_resilient(
    path: Path,
    *,
    max_retries: int = RESILIENT_IO_MAX_RETRIES,
    retry_delay_sec: float = RESILIENT_IO_RETRY_DELAY_SEC,
    max_retry_delay_sec: float = RESILIENT_IO_MAX_RETRY_DELAY_SEC,
) -> int:
    return sum(
        1
        for _ in iter_nonempty_lines_resilient(
            path,
            max_retries=max_retries,
            retry_delay_sec=retry_delay_sec,
            max_retry_delay_sec=max_retry_delay_sec,
        )
    )


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_token_budget_cache() -> Dict[str, Any]:
    if not TOKEN_BUDGET_CACHE_PATH.exists():
        return {}
    try:
        with TOKEN_BUDGET_CACHE_PATH.open("r", encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_token_budget_cache(payload: Dict[str, Any]) -> None:
    TOKEN_BUDGET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOKEN_BUDGET_CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_files(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.name).encode("utf-8"))
        if path.exists():
            h.update(sha256_file(path).encode("utf-8"))
    return h.hexdigest()


def fingerprint_metric_spec_metrics(metric_names: Sequence[str], metric_spec_path: Path = METRIC_SPEC_PATH) -> str:
    spec = load_json(metric_spec_path)
    metrics = spec.get("metrics") or {}
    selected = {
        name: metrics.get(name)
        for name in metric_names
    }
    payload = {
        "schema_version": spec.get("schema_version"),
        "metric_names": list(metric_names),
        "metrics": selected,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def scoring_metric_spec_fingerprint(metric_spec_path: Path = METRIC_SPEC_PATH) -> str:
    return fingerprint_metric_spec_metrics(
        CORE_SELECTION_METRICS + DIAGNOSTIC_METRICS,
        metric_spec_path=metric_spec_path,
    )


def normalize_dataset_config(config_path: Path) -> List[Dict[str, Any]]:
    payload = load_json(config_path)
    raw_specs = payload.get("datasets", []) if isinstance(payload, dict) else payload
    specs: List[Dict[str, Any]] = []
    for i, raw in enumerate(raw_specs):
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset spec #{i + 1} must be an object.")
        name = str(raw.get("name") or f"dataset_{i+1}").strip()
        fmt = str(raw.get("format") or "").strip()
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format for {name}: {fmt}")
        source = Path(str(raw.get("source") or "")).expanduser()
        if not source.is_absolute():
            source = (PROJECT_DIR / source).resolve()
        output_file = str(raw.get("output_file") or f"{name}_scored.jsonl")
        specs.append(
            {
                "name": name,
                "format": fmt,
                "source": source,
                "batch_glob": str(raw.get("batch_glob") or "batch_*.json"),
                "text_field": str(raw.get("text_field") or "text"),
                "id_fields": [str(x) for x in raw.get("id_fields", [])],
                "metadata_fields": [str(x) for x in raw.get("metadata_fields", [])],
                "min_text_chars": int(raw.get("min_text_chars", 50)),
                "output_file": output_file,
            }
        )
    return specs


def _get_tokenizer(tokenizer_name: str = DEFAULT_TOKENIZER_NAME):
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is not None:
        return tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception:
        tokenizer = None
    _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def estimate_token_count(text: str, tokenizer_name: str = DEFAULT_TOKENIZER_NAME) -> int:
    tokenizer = _get_tokenizer(tokenizer_name)
    if tokenizer is None:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])


def _dataset_source_signature(spec: Dict[str, Any]) -> Dict[str, Any]:
    source = Path(spec["source"])
    if spec["format"] == "json_batch_dir":
        files = sorted(source.glob(spec["batch_glob"]))
    else:
        files = [source]
    total_size = 0
    latest_mtime_ns = 0
    for path in files:
        if not path.exists():
            continue
        stat = path.stat()
        total_size += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
    return {
        "format": spec["format"],
        "source": str(source),
        "file_count": len(files),
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "batch_glob": spec.get("batch_glob"),
        "min_text_chars": int(spec.get("min_text_chars") or 0),
        "text_field": spec.get("text_field"),
    }


def dataset_token_budget(
    spec: Dict[str, Any],
    tokenizer_name: str = DEFAULT_TOKENIZER_NAME,
    refresh: bool = False,
) -> Dict[str, Any]:
    signature = _dataset_source_signature(spec)
    cache_key = json.dumps(
        {
            "dataset": spec["name"],
            "tokenizer_name": tokenizer_name,
            "signature": signature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    cache = _load_token_budget_cache()
    if not refresh:
        cached = cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

    total_tokens = 0
    total_docs = 0
    total_chars = 0
    text_field = spec["text_field"]
    min_text_chars = int(spec["min_text_chars"])
    for row in iter_documents(spec):
        doc = row["doc"]
        raw_text = str(doc.get(text_field) or "")
        stripped = raw_text.strip()
        if len(stripped) < min_text_chars:
            continue
        total_tokens += estimate_token_count(stripped, tokenizer_name=tokenizer_name)
        total_docs += 1
        total_chars += len(stripped)

    result = {
        "dataset": spec["name"],
        "tokenizer_name": tokenizer_name,
        "token_count": total_tokens,
        "document_count": total_docs,
        "character_count": total_chars,
        "signature": signature,
    }
    cache[cache_key] = result
    _save_token_budget_cache(cache)
    return result


def _stable_doc_id(doc: Dict[str, Any], doc_idx: int, dataset_name: str, id_fields: Sequence[str], text_field: str) -> str:
    for key in id_fields:
        raw = str(doc.get(key) or "").strip()
        if raw:
            return raw
    title = str(doc.get("title") or "").strip()
    text = str(doc.get(text_field) or "").strip()
    seed = f"{title}\n{text[:500]}"
    suffix = hashlib.md5(seed.encode("utf-8")).hexdigest()[:12] if seed else f"{doc_idx:08d}"
    return f"{dataset_name}_{doc_idx}_{suffix}"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0
    for para in paragraphs or [text]:
        words = para.split()
        if not words:
            continue
        if current and current_words + len(words) > chunk_size:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(para.replace("\n", " ").strip())
        current_words += len(words)
    if current:
        chunks.append(" ".join(current).strip())
    return [c for c in chunks if len(c.split()) >= MIN_CHUNK_WORDS]


def build_chunk_uid(dataset: str, source: str, doc_id: str, chunk_id: int) -> str:
    return f"{dataset}::{source}::{doc_id}::{chunk_id}"


def dedupe_doc_id(base_id: str, seen: Dict[str, int]) -> str:
    n = seen.get(base_id, 0) + 1
    seen[base_id] = n
    if n == 1:
        return base_id
    return f"{base_id}__dup{n}"


def iter_documents(spec: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    text_field = spec["text_field"]
    source = spec["source"]
    if spec["format"] == "json_list":
        payload = load_json(source)
        if not isinstance(payload, list):
            raise ValueError(f"{source} must contain a top-level list.")
        for idx, doc in enumerate(payload):
            if isinstance(doc, dict):
                yield {"doc": doc, "doc_idx": idx, "batch_file": ""}
    elif spec["format"] == "json_batch_dir":
        batch_files = sorted(source.glob(spec["batch_glob"]))
        for batch_file in batch_files:
            payload = load_json(batch_file)
            if not isinstance(payload, list):
                raise ValueError(f"{batch_file} must contain a top-level list.")
            for idx, doc in enumerate(payload):
                if isinstance(doc, dict):
                    yield {"doc": doc, "doc_idx": idx, "batch_file": batch_file.name}
    else:
        raise ValueError(f"Unsupported dataset format: {spec['format']}")


def iter_chunks(specs: Sequence[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for spec in specs:
        text_field = spec["text_field"]
        min_text_chars = int(spec["min_text_chars"])
        seen_doc_ids: Dict[str, int] = {}
        for row in iter_documents(spec):
            doc = row["doc"]
            raw_text = str(doc.get(text_field) or "")
            if len(raw_text.strip()) < min_text_chars:
                continue
            base_doc_id = _stable_doc_id(doc, int(row["doc_idx"]), spec["name"], spec["id_fields"], text_field)
            source = str(row.get("batch_file") or spec["name"])
            doc_id = dedupe_doc_id(f"{source}::{base_doc_id}", seen_doc_ids)
            metadata = {k: doc.get(k) for k in spec["metadata_fields"]}
            for chunk_id, chunk in enumerate(chunk_text(raw_text)):
                yield {
                    "dataset": spec["name"],
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_uid": build_chunk_uid(spec["name"], source, doc_id, chunk_id),
                    "text": chunk,
                    "word_count": len(chunk.split()),
                    "metadata": metadata,
                    "input_source": str(spec["source"]),
                }


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sentence_count(text: str) -> int:
    pieces = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    return len(pieces)


def alpha_ratio(text: str) -> float:
    chars = len(text)
    if chars == 0:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha() or ch.isspace())
    return alpha / chars


def repeated_token_ratio(text: str) -> float:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 1.0
    unique = len(set(tokens))
    return 1.0 - (unique / len(tokens))


def simhash64(text: str) -> int:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit, value in enumerate(weights):
        if value >= 0:
            out |= (1 << bit)
    return out


def simhash_prefix(value: int, bits: int = 16) -> str:
    shift = max(0, 64 - bits)
    return f"{(value >> shift):0{bits // 4}x}"


def token_ngram_jaccard(a_text: str, b_text: str, n: int = 3) -> float:
    def grams(text: str) -> set[str]:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < n:
            return set()
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    a = grams(a_text)
    b = grams(b_text)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
