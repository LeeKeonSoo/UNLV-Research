#!/usr/bin/env python3
"""Phase-1 artifact validation utilities.

This module is used by both manual validation runs and the orchestrator.
The default behavior is strict: if required files are missing or malformed,
validation fails with explicit reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
import types


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
KHAN_INPUT = PROJECT_DIR / "khan_k12_concepts" / "all_k12_concepts.json"
TINY_INPUT = PROJECT_DIR / "tiny_textbooks_raw"
KHAN_ANALYSIS = OUTPUT_DIR / "khan_analysis.jsonl"
TINY_ANALYSIS = OUTPUT_DIR / "tiny_textbooks_analysis.jsonl"
RUN_MANIFEST = OUTPUT_DIR / "run_manifest.json"
RUN_SUMMARY = OUTPUT_DIR / "run_summary.json"
CONCEPT_PROTOTYPES = OUTPUT_DIR / "concept_prototypes_tfidf.pkl"
KAHN_TAXONOMY = OUTPUT_DIR / "khan_taxonomy.json"
METADATA_PATH = OUTPUT_DIR / "metadata.json"
CORPUS_INDEX = OUTPUT_DIR / "corpus_index.pkl"
DASHBOARD_HTML = OUTPUT_DIR / "dashboard.html"

SCHEMA_VERSION = "v2"
REQUIRED_TIER_KEYS = {
    "domain",
    "quality",
    "difficulty",
    "redundancy",
    "perplexity",
}
REQUIRED_FLAGS = {
    "domain_valid",
    "quality_valid",
    "difficulty_valid",
    "redundancy_valid",
    "perplexity_valid",
}
ALLOWED_TIER_VALUES = {"core", "exploratory"}


@dataclass
class ValidationItem:
    name: str
    ok: bool
    message: str
    details: Dict[str, Any]


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl_lines(path: Path, limit: Optional[int] = None) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _iter_jsonl_records(path: Path, limit: Optional[int] = None):
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)
            count += 1
            if limit is not None and count >= limit:
                return


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(ok: bool, name: str, details: Dict[str, Any], failures: List[ValidationItem]):
    if not ok:
        failures.append(ValidationItem(name=name, ok=False, message="FAIL", details=details))
    return ok


def validate_khan_collection(
    path: Path = KHAN_INPUT,
    sample_only: bool = False,
    max_records: Optional[int] = None,
) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not path.exists():
        _require(False, "khan_collection_file", {"path": str(path)}, failures)
        return failures

    try:
        data = _load_json(path)
    except Exception as e:
        _require(False, "khan_collection_json", {"path": str(path), "error": str(e)}, failures)
        return failures

    if not isinstance(data, list):
        _require(False, "khan_collection_type", {"type": type(data).__name__}, failures)
        return failures

    if len(data) == 0:
        _require(False, "khan_collection_nonempty", {"path": str(path), "count": 0}, failures)
        return failures

    records = data if max_records is None else data[:max_records]
    expected_fields = {"subject", "grade", "title", "content"}
    missing_field_rows = []
    missing_doc_ids = 0
    empty_content = 0
    duplicate_doc = 0
    seen_doc_ids = set()
    for i, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            missing_field_rows.append((i, "non_object"))
            continue
        if not expected_fields.issubset(rec.keys()):
            missing = sorted(expected_fields - set(rec.keys()))
            missing_field_rows.append((i, ",".join(missing)))
        if not isinstance(rec.get("content", ""), str) or len(rec.get("content", "").strip()) < 50:
            empty_content += 1
        doc_id = rec.get("doc_id") or rec.get("id")
        if not doc_id:
            if rec.get("url") and rec.get("title"):
                doc_id = f"{rec['url']}::{rec['title']}"
            elif rec.get("url"):
                doc_id = f"{rec['url']}::{i}"
            elif rec.get("title"):
                doc_id = f"title::{rec['title']}"

        if doc_id and isinstance(rec.get("content"), str):
            doc_id = f"{doc_id}::{hashlib.md5(rec['content'].encode('utf-8')).hexdigest()[:12]}"
        if not doc_id:
            missing_doc_ids += 1
            continue
        if doc_id in seen_doc_ids:
            duplicate_doc += 1
        else:
            seen_doc_ids.add(doc_id)

    if missing_field_rows:
        _require(False, "khan_collection_required_fields", {
            "sample_records": len(records),
            "missing_examples": missing_field_rows[:10],
        }, failures)

    _require(
        empty_content < len(records) * 0.10,
        "khan_collection_content",
        {
            "records": len(records),
            "short_content": empty_content,
            "sample_limit": max_records,
        },
        failures,
    )

    _require(
        missing_doc_ids == 0,
        "khan_collection_missing_doc_id",
        {
            "checked_records": len(records),
            "missing_doc_ids": missing_doc_ids,
        },
        failures,
    )

    _require(
        duplicate_doc == 0,
        "khan_collection_unique_doc_id",
        {
            "checked_records": len(records),
            "duplicate_doc_ids": duplicate_doc,
        },
        failures,
    )

    if not failures:
        failures.append(ValidationItem(
            name="khan_collection",
            ok=True,
            message=f"PASS ({len(data)} docs checked, sample_limit={max_records}, sample_only={sample_only})",
            details={
                "path": str(path),
                "docs_in_file": len(data),
            },
        ))
    return failures


def validate_tiny_collection(
    directory: Path = TINY_INPUT,
    max_files: Optional[int] = None,
) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not directory.exists():
        failures.append(ValidationItem(
            name="tiny_collection_dir",
            ok=False,
            message="FAIL",
            details={"path": str(directory)},
        ))
        return failures

    batch_files = sorted(directory.glob("batch_*.json"))
    if max_files:
        batch_files = batch_files[:max_files]

    if not batch_files:
        failures.append(ValidationItem(
            name="tiny_collection_batches",
            ok=False,
            message="FAIL",
            details={"path": str(directory)},
        ))
        return failures

    docs = 0
    bad_files = []
    short_or_empty = 0
    missing_fields = 0

    for bf in batch_files:
        try:
            data = _load_json(bf)
        except Exception as e:
            bad_files.append({"file": bf.name, "error": str(e)})
            continue

        if not isinstance(data, list):
            bad_files.append({"file": bf.name, "error": "not a list"})
            continue

        if not data:
            bad_files.append({"file": bf.name, "error": "empty batch"})
            continue

        for item in data:
            docs += 1
            if not isinstance(item, dict):
                missing_fields += 1
                continue
            if not str(item.get("id") or item.get("text") or ""):
                missing_fields += 1
            text = item.get("text", "")
            if not isinstance(text, str) or len(text.strip()) < 100:
                short_or_empty += 1

    _require(len(bad_files) == 0, "tiny_collection_batch_health", {
        "bad_files": bad_files,
        "checked_batches": len(batch_files),
    }, failures)

    if bad_files:
        return failures

    _require(short_or_empty == 0, "tiny_collection_text_length", {
        "checked_docs": docs,
        "short_or_empty": short_or_empty,
    }, failures)

    _require(missing_fields == 0, "tiny_collection_required_fields", {
        "checked_docs": docs,
        "missing_fields": missing_fields,
    }, failures)

    if not failures:
        failures.append(ValidationItem(
            name="tiny_collection",
            ok=True,
            message="PASS",
            details={
                "checked_batches": len(batch_files),
                "batch_files": [p.name for p in batch_files],
                "docs_checked": docs,
            },
        ))
    return failures


def _validate_metric_record(record: Dict[str, Any], index: int, source: str, details: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    """Return counts: bad_schema, bad_tier, bad_flags, bad_ranges, bad_source."""
    bad_schema = bad_tier = bad_flags = bad_ranges = bad_source = 0

    if record.get("schema_version") != SCHEMA_VERSION:
        bad_schema += 1

    if record.get("source") != source:
        bad_source += 1

    metric_tier = record.get("metric_tier") or {}
    for key in REQUIRED_TIER_KEYS:
        if key not in metric_tier or metric_tier.get(key) not in ALLOWED_TIER_VALUES:
            bad_tier += 1

    validity_flags = record.get("validity_flags") or {}
    for key in REQUIRED_FLAGS:
        if key not in validity_flags or not _is_bool(validity_flags.get(key)):
            bad_flags += 1

    # Basic field shape checks
    if not isinstance(record.get("text"), str) or not record.get("text"):
        bad_ranges += 1
    if not isinstance(record.get("chunk_id"), int) or record.get("chunk_id") < 0:
        bad_ranges += 1

    domain_labels = record.get("domain_labels") or {}
    if not isinstance(domain_labels, dict):
        bad_ranges += 1
    else:
        vals = [_safe_float(v) for v in domain_labels.values()]
        vals = [v for v in vals if v is not None]
        bad_ranges += sum(1 for v in vals if v < 0 or v > 1.0)

    quality_score = _safe_float(record.get("quality_score"))
    if quality_score is None or not (0.0 <= quality_score <= 1.0):
        bad_ranges += 1

    markers = record.get("educational_markers") or {}
    if not isinstance(markers, dict):
        bad_ranges += 1
    else:
        for k in ("has_examples", "has_explanation", "has_structure"):
            if not _is_bool(markers.get(k)):
                bad_ranges += 1

    difficulty = record.get("difficulty") or {}
    for key in ("flesch_kincaid_grade", "flesch_reading_ease", "smog_index", "lexical_diversity", "rare_words_pct"):
        if key not in difficulty:
            bad_ranges += 1
            continue
        v = _safe_float(difficulty.get(key))
        if v is None:
            bad_ranges += 1

    return bad_schema, bad_tier, bad_flags, bad_ranges, bad_source


def _validate_analysis_jsonl(path: Path, source: str, max_records: Optional[int] = None) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not path.exists():
        failures.append(ValidationItem(name=f"analysis_{source}", ok=False, message="FAIL", details={"path": str(path)}))
        return failures

    bad_schema = bad_tier = bad_flags = bad_ranges = bad_source = 0
    total = 0
    for idx, rec in enumerate(_iter_jsonl_records(path, limit=max_records), 1):
        if not isinstance(rec, dict):
            bad_ranges += 1
            continue

        total = idx
        rec_bad_schema, rec_bad_tier, rec_bad_flags, rec_bad_ranges, rec_bad_source = _validate_metric_record(
            rec,
            idx,
            source,
            {},
        )
        bad_schema += rec_bad_schema
        bad_tier += rec_bad_tier
        bad_flags += rec_bad_flags
        bad_ranges += rec_bad_ranges
        bad_source += rec_bad_source

    if total == 0:
        failures.append(ValidationItem(
            name=f"analysis_{source}",
            ok=False,
            message="FAIL",
            details={"path": str(path), "records": 0},
        ))
        return failures

    if max_records is not None and total >= max_records:
        details = {"path": str(path), "records": total, "sample_only": True}
    else:
        details = {"path": str(path), "records": total, "sample_only": False}

    if bad_schema:
        details["bad_schema"] = bad_schema
    if bad_tier:
        details["bad_tier"] = bad_tier
    if bad_flags:
        details["bad_flags"] = bad_flags
    if bad_ranges:
        details["bad_ranges"] = bad_ranges
    if bad_source:
        details["bad_source"] = bad_source

    if bad_schema or bad_tier or bad_flags or bad_ranges or bad_source:
        details["ok"] = False
        failures.append(ValidationItem(name=f"analysis_{source}_record_schema", ok=False, message="FAIL", details=details))
        return failures

    failures.append(ValidationItem(name=f"analysis_{source}", ok=True, message="PASS", details=details))
    return failures


def validate_analysis_outputs(
    khan_path: Path = KHAN_ANALYSIS,
    tiny_path: Path = TINY_ANALYSIS,
    max_records: Optional[int] = None,
) -> List[ValidationItem]:
    results: List[ValidationItem] = []
    results.extend(_validate_analysis_jsonl(khan_path, "khan_academy", max_records=max_records))
    results.extend(_validate_analysis_jsonl(tiny_path, "tiny_textbooks", max_records=max_records))
    return results


def _register_datasketch_fallback() -> None:
    """Register lightweight datasketch stubs so legacy pickle payloads can load."""
    if "datasketch" in sys.modules and "datasketch.hashfunc" in sys.modules:
        return

    datasketch_module = types.ModuleType("datasketch")
    minhash_module = types.ModuleType("datasketch.minhash")
    lsh_module = types.ModuleType("datasketch.lsh")
    storage_module = types.ModuleType("datasketch.storage")
    hashfunc_module = types.ModuleType("datasketch.hashfunc")

    class _FallbackMinHash:
        def __init__(self, *args, **kwargs):
            self._hashvalues = []
            self.num_perm = kwargs.get("num_perm", 128) if kwargs else 128

        def update(self, *_args, **_kwargs):
            return None

        def jaccard(self, _other):
            return 0.0

        def _byteswap(self, value=None):
            if value is None:
                return self
            return value

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

        def __getstate__(self):
            return self.__dict__

    class _FallbackLSH:
        def __init__(self, *args, **kwargs):
            self._threshold = kwargs.get("threshold", 0.5)
            self._num_perm = kwargs.get("num_perm", 128)
            self._lsh = {}
            self._minhash_param = kwargs.copy() if hasattr(kwargs, "copy") else {}

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            elif isinstance(state, (list, tuple)) and state:
                # Handle legacy pickle formats that store a tuple of state values.
                self.__dict__["_fallback_state"] = state

        def __getstate__(self):
            return self.__dict__

        def _byteswap(self, value):
            return value

        def query(self, *_args, **_kwargs):
            return []

        def insert(self, *_args, **_kwargs):
            return None

    class _FallbackStorage:
        def __init__(self, *args, **kwargs):
            self._data = {}

        def add(self, *_args, **_kwargs):
            return None

        def get(self, *_args, **_kwargs):
            return []

    def _hash_func(*_args, **_kwargs):
        return 0

    def _missing_storage_attribute(name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return _FallbackStorage

    def _missing_lsh_attribute(name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return _FallbackLSH

    minhash_module.MinHash = _FallbackMinHash
    minhash_module.MinHashLSH = _FallbackLSH
    datasketch_module.MinHash = _FallbackMinHash
    datasketch_module.MinHashLSH = _FallbackLSH
    lsh_module.MinHashLSH = _FallbackLSH
    storage_module.MinHashLSH = _FallbackLSH
    storage_module.MinHash = _FallbackMinHash
    storage_module.DictSetStorage = _FallbackStorage
    lsh_module.__getattr__ = _missing_lsh_attribute
    storage_module.__getattr__ = _missing_storage_attribute
    hashfunc_module.sha1_hash32 = _hash_func

    sys.modules["datasketch"] = datasketch_module
    sys.modules["datasketch.minhash"] = minhash_module
    sys.modules["datasketch.lsh"] = lsh_module
    sys.modules["datasketch.storage"] = storage_module
    sys.modules["datasketch.hashfunc"] = hashfunc_module


def _safe_pickle_load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        _register_datasketch_fallback()
        if "Can't get attribute" in str(e) or "No module named 'datasketch'" in str(e):
            with path.open("rb") as f:
                return pickle.load(f)
        raise


def validate_taxonomy_outputs(
    taxonomy_path: Path = KAHN_TAXONOMY,
    metadata_path: Path = METADATA_PATH,
    pkl_path: Path = CONCEPT_PROTOTYPES,
) -> List[ValidationItem]:
    failures: List[ValidationItem] = []

    if not taxonomy_path.exists() or not metadata_path.exists() or not pkl_path.exists():
        bad = {
            "khan_taxonomy": taxonomy_path.exists(),
            "metadata": metadata_path.exists(),
            "prototypes": pkl_path.exists(),
        }
        failures.append(ValidationItem(name="taxonomy_outputs", ok=False, message="FAIL", details=bad))
        return failures

    try:
        tax = _load_json(taxonomy_path)
        if not isinstance(tax, dict) or not tax:
            failures.append(ValidationItem(name="khan_taxonomy_format", ok=False, message="FAIL", details={"path": str(taxonomy_path)}))
    except Exception as e:
        failures.append(ValidationItem(name="khan_taxonomy_format", ok=False, message="FAIL", details={"error": str(e)}))
        tax = {}

    try:
        meta = _load_json(metadata_path)
        if not isinstance(meta, dict) or not meta:
            failures.append(ValidationItem(name="khan_metadata", ok=False, message="FAIL", details={"path": str(metadata_path)}))
    except Exception as e:
        failures.append(ValidationItem(name="khan_metadata", ok=False, message="FAIL", details={"error": str(e)}))
        meta = {}

    try:
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        prototypes = obj.get("prototypes") if isinstance(obj, dict) else None
        vectorizer = obj.get("vectorizer") if isinstance(obj, dict) else None
        if not isinstance(prototypes, dict) or not prototypes:
            failures.append(ValidationItem(name="khan_prototypes", ok=False, message="FAIL", details={"path": str(pkl_path)}))
        if vectorizer is None:
            failures.append(ValidationItem(name="khan_vectorizer", ok=False, message="FAIL", details={"path": str(pkl_path)}))
    except Exception as e:
        failures.append(ValidationItem(name="khan_prototypes_pickle", ok=False, message="FAIL", details={"path": str(pkl_path), "error": str(e)}))

    if failures:
        return failures

    failures.append(ValidationItem(
        name="taxonomy_outputs",
        ok=True,
        message="PASS",
        details={
            "courses": len(tax),
            "prototype_count": len(prototypes),
            "vectorizer": bool(vectorizer),
            "metadata_keys": sorted(meta.keys()),
        },
    ))
    return failures


def validate_corpus_index(path: Path = CORPUS_INDEX) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not path.exists():
        failures.append(ValidationItem(name="corpus_index_exists", ok=False, message="FAIL", details={"path": str(path)}))
        return failures

    try:
        idx = _safe_pickle_load(path)
    except Exception as e:
        failures.append(ValidationItem(name="corpus_index_load", ok=False, message="FAIL", details={"error": str(e)}))
        return failures

    required = {"doc_ids", "doc_hash_by_id", "exact_hash_counts"}
    missing = sorted(required - set(idx.keys())) if isinstance(idx, dict) else ["invalid_type"]
    if missing:
        failures.append(ValidationItem(name="corpus_index_schema", ok=False, message="FAIL", details={"missing_keys": missing}))
        return failures

    doc_ids = idx.get("doc_ids") or []
    doc_hash_by_id = idx.get("doc_hash_by_id") or {}

    if not isinstance(doc_ids, list) or not doc_ids:
        failures.append(ValidationItem(name="corpus_index_docs", ok=False, message="FAIL", details={"doc_ids_type": type(doc_ids).__name__}))

    if not isinstance(doc_hash_by_id, dict) or not doc_hash_by_id:
        failures.append(ValidationItem(name="corpus_index_hash_map", ok=False, message="FAIL", details={"doc_hash_by_id_type": type(doc_hash_by_id).__name__}))
    else:
        if len(doc_hash_by_id) != len(doc_ids):
            failures.append(ValidationItem(name="corpus_index_counts", ok=False, message="FAIL", details={
                "doc_ids": len(doc_ids),
                "doc_hash_by_id": len(doc_hash_by_id),
            }))

        missing_ids = sum(1 for k in doc_ids if k not in doc_hash_by_id)
        if missing_ids:
            failures.append(ValidationItem(name="corpus_index_id_coverage", ok=False, message="FAIL", details={"missing_ids": missing_ids}))

    if failures:
        return failures

    if idx.get("BUILD_TFIDF_MATRIX", True) is False:  # backward-compatible flag in old payloads
        tfidf_state = "disabled"
    else:
        tfidf_state = "enabled"

    failures.append(ValidationItem(
        name="corpus_index",
        ok=True,
        message="PASS",
        details={
            "doc_count": len(doc_ids),
            "exact_hashes": len(idx.get("exact_hash_counts", {})),
            "lsh_enabled": idx.get("lsh") is not None,
            "minhashes": len(idx.get("minhashes") or {}),
            "tfidf_state": tfidf_state,
            "doc_texts_stored": bool(idx.get("doc_texts")),
        },
    ))
    return failures


def validate_manifest(
    manifest_path: Path = RUN_MANIFEST,
    run_summary_path: Path = RUN_SUMMARY,
    khan_output: Path = KHAN_ANALYSIS,
    tiny_output: Path = TINY_ANALYSIS,
    require_gates: bool = False,
) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not manifest_path.exists():
        failures.append(ValidationItem(name="run_manifest_exists", ok=False, message="FAIL", details={"path": str(manifest_path)}))
        return failures

    try:
        manifest = _load_json(manifest_path)
    except Exception as e:
        failures.append(ValidationItem(name="run_manifest_json", ok=False, message="FAIL", details={"error": str(e)}))
        return failures

    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append(ValidationItem(name="run_manifest_schema", ok=False, message="FAIL", details={
            "schema_version": manifest.get("schema_version") if isinstance(manifest, dict) else None,
        }))

    required = {
        "dataset_versions_and_counts",
        "threshold_config",
        "runtime_device",
        "metric_tier",
        "reliability_gate_outcomes",
    }
    if isinstance(manifest, dict):
        missing = sorted(required - set(manifest.keys()))
        if missing:
            failures.append(ValidationItem(name="run_manifest_keys", ok=False, message="FAIL", details={"missing_keys": missing}))

    if isinstance(manifest, dict):
        khan_counts = manifest.get("dataset_versions_and_counts", {}).get("khan_academy", {})
        tiny_counts = manifest.get("dataset_versions_and_counts", {}).get("tiny_textbooks", {})
        actual_k = _count_jsonl_lines(khan_output)
        actual_t = _count_jsonl_lines(tiny_output)

        if khan_counts.get("chunks_written") != actual_k:
            failures.append(ValidationItem(name="run_manifest_khan_count", ok=False, message="FAIL", details={
                "manifest": khan_counts.get("chunks_written"),
                "actual": actual_k,
            }))
        if tiny_counts.get("chunks_written") != actual_t:
            failures.append(ValidationItem(name="run_manifest_tiny_count", ok=False, message="FAIL", details={
                "manifest": tiny_counts.get("chunks_written"),
                "actual": actual_t,
            }))

        for metric in REQUIRED_TIER_KEYS:
            if manifest.get("metric_tier", {}).get(metric) not in ALLOWED_TIER_VALUES:
                failures.append(ValidationItem(name="run_manifest_metric_tier", ok=False, message="FAIL", details={"metric": metric}))

        gate_outcomes = manifest.get("reliability_gate_outcomes", {})
        if not isinstance(gate_outcomes, dict) or not gate_outcomes:
            failures.append(ValidationItem(name="run_manifest_gates", ok=False, message="FAIL", details={"reliability_gate_outcomes": gate_outcomes}))
        elif require_gates:
            for metric, outcome in gate_outcomes.items():
                if not isinstance(outcome, dict):
                    continue
                for _, metric_state in outcome.items():
                    if isinstance(metric_state, dict) and "pass" in metric_state:
                        if metric_state.get("pass") is False:
                            failures.append(ValidationItem(name="run_manifest_gate_threshold", ok=False, message="FAIL", details={
                                "metric": metric,
                                "metric_state": metric_state,
                            }))

    if run_summary_path.exists():
        try:
            summary = _load_json(run_summary_path)
            if not isinstance(summary, dict):
                failures.append(ValidationItem(name="run_summary_format", ok=False, message="FAIL", details={"path": str(run_summary_path)}))
        except Exception as e:
            failures.append(ValidationItem(name="run_summary_json", ok=False, message="FAIL", details={"error": str(e)}))

    if not failures:
        failures.append(ValidationItem(
            name="run_manifest",
            ok=True,
            message="PASS",
            details={
                "schema_version": manifest.get("schema_version"),
                "khan_chunks": manifest.get("dataset_versions_and_counts", {}).get("khan_academy", {}).get("chunks_written"),
                "tiny_chunks": manifest.get("dataset_versions_and_counts", {}).get("tiny_textbooks", {}).get("chunks_written"),
                "device": manifest.get("runtime_device", {}),
            },
        ))
    return failures


def validate_dashboard(path: Path = DASHBOARD_HTML, manifest_path: Path = RUN_MANIFEST) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not path.exists():
        failures.append(ValidationItem(name="dashboard_exists", ok=False, message="FAIL", details={"path": str(path)}))
        return failures

    text = path.read_text(encoding="utf-8", errors="replace")
    if "Dataset Analysis Dashboard" not in text:
        failures.append(ValidationItem(name="dashboard_title", ok=False, message="FAIL", details={"path": str(path)}))

    if "const EMB =" not in text:
        failures.append(ValidationItem(name="dashboard_embedded_data", ok=False, message="FAIL", details={"path": str(path)}))

    emb_match = re.search(
        r"const\s+EMB\s*=\s*(\{.*?\});\s*\n\s*const\s+KS\s*=",
        text,
        flags=re.DOTALL,
    )
    if not emb_match:
        failures.append(ValidationItem(
            name="dashboard_embedded_data",
            ok=False,
            message="FAIL",
            details={"error": "cannot locate embedded data boundary"},
        ))
    else:
        try:
            emb = json.loads(emb_match.group(1))
            if not isinstance(emb, dict) or not emb.get("khan") or not emb.get("tiny"):
                failures.append(ValidationItem(name="dashboard_embedded_data", ok=False, message="FAIL", details={"embedded_shape": type(emb).__name__}))
            else:
                khan_stats = (emb.get("khan", {}) or {}).get("stats", {})
                tiny_stats = (emb.get("tiny", {}) or {}).get("stats", {})
                khan_total = khan_stats.get("total")
                tiny_total = tiny_stats.get("total")
                if not isinstance(khan_total, int) or not isinstance(tiny_total, int):
                    failures.append(ValidationItem(name="dashboard_stats", ok=False, message="FAIL", details={"khan_total": khan_total, "tiny_total": tiny_total}))
        except Exception as e:
            failures.append(ValidationItem(name="dashboard_embedded_data", ok=False, message="FAIL", details={"error": str(e)}))

    if failures:
        return failures

    details = {"path": str(path)}
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        details["manifest_schema_version"] = manifest.get("schema_version") if isinstance(manifest, dict) else None

    failures.append(ValidationItem(name="dashboard", ok=True, message="PASS", details=details))
    return failures


def summarize(results: List[ValidationItem]) -> Tuple[bool, Dict[str, Any]]:
    failed = [r for r in results if not r.ok]
    passed = [r for r in results if r.ok]
    return len(failed) == 0, {
        "passed": len(passed),
        "failed": len(failed),
        "total": len(results),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "failed_items": [
            {"name": r.name, "message": r.message, "details": r.details}
            for r in failed
        ],
    }


def validate_all(
    profile_only: str = "full",
    max_tiny_files: Optional[int] = None,
    max_metrics_records: Optional[int] = None,
    require_gates: bool = False,
    sample_only: bool = False,
) -> Tuple[bool, List[ValidationItem]]:
    results: List[ValidationItem] = []

    if profile_only in {"full", "collections", "collections_only"}:
        results.extend(validate_khan_collection(sample_only=sample_only, max_records=max_tiny_files))
        results.extend(validate_tiny_collection(max_files=max_tiny_files))

    if profile_only in {"full", "taxonomy", "index"}:
        results.extend(validate_taxonomy_outputs())

    if profile_only in {"full", "index"}:
        results.extend(validate_corpus_index())

    if profile_only in {"full", "metrics"}:
        results.extend(validate_analysis_outputs(max_records=max_metrics_records))
        results.extend(validate_manifest(require_gates=require_gates))
        results.extend(validate_dashboard())

    ok = all(r.ok for r in results)
    return ok, results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase-1 artifacts")
    parser.add_argument(
        "--profile",
        choices={"full", "collections", "taxonomy", "index", "metrics"},
        default="full",
        help="Which artifact group to validate.",
    )
    parser.add_argument(
        "--max-tiny-files",
        type=int,
        default=None,
        help="Limit tiny batch files checked in tiny collection validation.",
    )
    parser.add_argument(
        "--max-metrics-records",
        type=int,
        default=None,
        help="Limit records checked per metrics JSONL file.",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Fail if any gate in run_manifest is pass==False.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Legacy alias for tiny-file limited collection checks.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Write validation report JSON to file.",
    )

    args = parser.parse_args()

    ok, results = validate_all(
        profile_only=args.profile,
        max_tiny_files=args.max_tiny_files,
        max_metrics_records=args.max_metrics_records,
        require_gates=args.require_gates,
        sample_only=args.sample_only,
    )
    pass_flag, summary_payload = summarize(results)

    print("Validation summary:")
    print(f"  total:   {summary_payload['total']}")
    print(f"  pass:    {summary_payload['passed']}")
    print(f"  fail:    {summary_payload['failed']}")

    if summary_payload["failed"]:
        print("\n[FAIL] First failures:")
        for item in summary_payload["failed_items"][:10]:
            print(f"  - {item['name']}: {item['message']} :: {item['details']}")

    if args.write_report:
        out_path = Path(args.write_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"summary": summary_payload, "results": [
                {
                    "name": r.name,
                    "ok": r.ok,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ]}, f, indent=2, ensure_ascii=False)

    return 0 if pass_flag else 1


if __name__ == "__main__":
    raise SystemExit(main())
