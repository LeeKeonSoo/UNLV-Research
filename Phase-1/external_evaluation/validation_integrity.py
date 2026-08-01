"""Validate record/text-disjoint, frozen-policy external evaluation inputs."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from curation_artifacts import load_json, sha256_file


JsonMap = dict[str, Any]


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _required_mapping(value: Any, name: str) -> JsonMap:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object.")
    return value


def _required_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty path.")
    return Path(value)


def _record_id(row: JsonMap, index: int) -> str | None:
    for field in ("record_id", "id", "uid"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _text(row: JsonMap) -> str:
    for field in ("text", "content", "document", "body"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    return ""


def _hashes(rows: Iterable[JsonMap]) -> tuple[set[str], set[str], int]:
    record_ids: set[str] = set()
    text_hashes: set[str] = set()
    missing_record_ids = 0
    for index, row in enumerate(rows):
        record_id = _record_id(row, index)
        if record_id is None:
            missing_record_ids += 1
        else:
            record_ids.add(record_id)
        normalized = _normalized(_text(row))
        if normalized:
            text_hashes.add(hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return record_ids, text_hashes, missing_record_ids


def _audited_candidate(report_path: Path) -> tuple[Path, JsonMap, JsonMap]:
    report = load_json(report_path)
    report_audit = _required_mapping(report.get("pretraining_audit"), f"{report_path} pretraining_audit")
    audit_path = _required_path(report_audit.get("path"), f"{report_path} pretraining_audit.path")
    expected_audit_sha = str(report_audit.get("sha256") or "")
    if expected_audit_sha != sha256_file(audit_path):
        raise RuntimeError(f"Pretraining audit hash mismatch in curation report: {report_path}")
    audit = load_json(audit_path)
    audited_output = _required_mapping(audit.get("audited_output"), f"{audit_path} audited_output")
    candidate_path = _required_path(audited_output.get("path"), f"{audit_path} audited_output.path")
    expected_candidate_sha = str(audited_output.get("sha256") or "")
    if expected_candidate_sha != sha256_file(candidate_path):
        raise RuntimeError(f"Audited candidate hash mismatch: {candidate_path}")
    return candidate_path, report, audit


def _audit_ready(audit: JsonMap) -> bool:
    return audit.get("status") == "benchmark_exclusion_complete" and audit.get("pretraining_eligible") is True


def build_validation_integrity_report(
    *,
    development_curation_report: Path,
    confirmatory_curation_report: Path,
) -> JsonMap:
    """Build a non-runtime gate for a record/text-disjoint confirmatory evaluation."""
    development_path, development_report, development_audit = _audited_candidate(development_curation_report)
    confirmatory_path, confirmatory_report, confirmatory_audit = _audited_candidate(confirmatory_curation_report)
    development_rows = _read_jsonl(development_path)
    confirmatory_rows = _read_jsonl(confirmatory_path)
    development_ids, development_texts, development_missing_ids = _hashes(development_rows)
    confirmatory_ids, confirmatory_texts, confirmatory_missing_ids = _hashes(confirmatory_rows)
    shared_ids = development_ids & confirmatory_ids
    shared_texts = development_texts & confirmatory_texts
    policy_match = development_report.get("policy_fingerprint") == confirmatory_report.get("policy_fingerprint")
    blocking_reasons: list[str] = []
    if not _audit_ready(development_audit):
        blocking_reasons.append("development_benchmark_exclusion_incomplete")
    if not _audit_ready(confirmatory_audit):
        blocking_reasons.append("confirmatory_benchmark_exclusion_incomplete")
    if development_missing_ids or confirmatory_missing_ids:
        blocking_reasons.append("stable_record_ids_required_for_corpus_disjointness")
    if shared_ids or shared_texts:
        blocking_reasons.append("development_and_confirmatory_corpora_overlap")
    if not policy_match:
        blocking_reasons.append("policy_fingerprint_mismatch")
    return {
        "schema_version": "external-validation-integrity-report-v1",
        "status": "confirmatory_ready" if not blocking_reasons else "confirmatory_blocked",
        "blocking_reasons": blocking_reasons,
        "development": {
            "curation_report": {"path": str(development_curation_report), "sha256": sha256_file(development_curation_report)},
            "audited_candidate": {"path": str(development_path), "sha256": sha256_file(development_path), "records": len(development_rows)},
        },
        "confirmatory": {
            "curation_report": {"path": str(confirmatory_curation_report), "sha256": sha256_file(confirmatory_curation_report)},
            "audited_candidate": {"path": str(confirmatory_path), "sha256": sha256_file(confirmatory_path), "records": len(confirmatory_rows)},
        },
        "corpus_disjointness": {
            "shared_record_id_count": len(shared_ids),
            "shared_normalized_text_count": len(shared_texts),
            "development_missing_record_id_count": development_missing_ids,
            "confirmatory_missing_record_id_count": confirmatory_missing_ids,
            "method": "stable_record_id_and_normalized_text_sha256_v1",
        },
        "policy_fingerprint_match": policy_match,
        "runtime_boundary": {
            "validation_metadata_visible_to_curation_runtime": False,
            "enforcement": "This report is built after both A-B-C materializations and is never loaded by run_curation.py.",
        },
        "claim_boundary": "This gate establishes input separation and frozen-policy identity only. It does not establish intrinsic data quality or downstream model benefit.",
    }
