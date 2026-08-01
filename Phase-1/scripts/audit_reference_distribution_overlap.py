#!/usr/bin/env python3
"""Audit whether source-role review records overlap existing Stage-A evidence."""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _mapping(row: JsonMap, key: str) -> JsonMap:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _evidence(record: JsonMap) -> JsonMap:
    eligibility = _mapping(record, "release_eligibility")
    rights = _mapping(record, "rights")
    quarantine = _mapping(record, "quarantine")
    hazards = _mapping(record, "hazards")
    composition = _mapping(record, "composition")
    language = _mapping(record, "language")
    no_known_hazard = all(
        hazards.get(field) is False
        for field in ("pii_detected", "secret_detected", "poisoning_suspected", "benchmark_contamination")
    )
    return {
        "release_eligible": eligibility.get("eligible") is True,
        "rights_allowed": rights.get("status") == "allowed",
        "release_candidate": quarantine.get("status") == "release_candidate",
        "no_known_hazard": no_known_hazard,
        "declared_code_domain": composition.get("content_domain") == "code",
        "declared_language": language.get("code"),
    }


def build_overlap_audit(review_sample: JsonMap, candidate_rows: Iterable[JsonMap]) -> JsonMap:
    """Compare review-only source-role records against immutable Stage-A evidence."""
    review_records = review_sample.get("review_records")
    if not isinstance(review_records, list):
        raise RuntimeError("Review sample must contain review_records.")
    candidates_by_id = {str(row.get("record_id") or ""): row for row in candidate_rows}
    results: list[JsonMap] = []
    for review in review_records:
        if not isinstance(review, dict):
            continue
        record_id = str(review.get("record_id") or "")
        candidate = candidates_by_id.get(record_id)
        if candidate is None:
            results.append({"record_id": record_id, "evidence_status": "missing_from_stage_a_input_needs_review"})
            continue
        evidence = _evidence(candidate)
        all_present = all(
            evidence[field]
            for field in ("release_eligible", "rights_allowed", "release_candidate", "no_known_hazard", "declared_code_domain")
        )
        results.append(
            {
                "record_id": record_id,
                "reference_distribution_score": review.get("reference_distribution_score"),
                **evidence,
                "evidence_status": "eligible_code_overlap_not_a_removal_decision" if all_present else "scope_or_admission_mismatch_needs_review",
            }
        )
    complete = [record for record in results if record.get("evidence_status") == "eligible_code_overlap_not_a_removal_decision"]
    return {
        "schema_version": "reference-distribution-overlap-audit-v1",
        "status": "overlap_audit_complete_not_a_selection_policy",
        "summary": {
            "review_records": len(results),
            "all_stage_a_evidence_present": len(complete),
            "needs_review": len(results) - len(complete),
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
            "selection_decisions_emitted": False,
            "data_removed": False,
        },
        "claim_boundary": "Existing Stage-A evidence can explain whether a reviewed record is already eligible. It cannot validate source-role score as a Quality measure or removal rule.",
        "records": results,
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit review-only source-role records against Stage-A evidence.")
    parser.add_argument("--review-sample", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_overlap_audit(json.loads(args.review_sample.read_text(encoding="utf-8")), _read_jsonl(args.candidate_input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
