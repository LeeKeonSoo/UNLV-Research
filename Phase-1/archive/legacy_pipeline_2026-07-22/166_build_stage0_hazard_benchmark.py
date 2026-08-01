#!/usr/bin/env python3
"""Build a labeled Stage-0 hazard fixture benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.normalize import process_candidate


DEFAULT_FIXTURES = Path("validation") / "fixtures" / "stage0_hazard_benchmark_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage0_hazard_benchmark_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage0_hazard_benchmark_report.md"


def _raw_case(case: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "id": str(case["id"]),
        "text": str(case.get("text") or ""),
        "pii_context": str(case.get("pii_context") or "general"),
        "source_name": "stage0-hazard-fixture",
        "source_uri": f"fixture://stage0-hazard/{case['id']}",
        "collected_at": "2026-06-23T00:00:00Z",
        "language": {"code": "en", "confidence": 1.0},
        "rights": case.get("rights") if isinstance(case.get("rights"), dict) else {"status": "unknown", "license": None},
    }
    for field in case.get("omit_provenance_fields") or []:
        record.pop(str(field), None)
    return record


def _evaluate_case(case: Dict[str, Any], index: int) -> Dict[str, Any]:
    record = process_candidate(_raw_case(case), index=index)
    blockers: List[str] = []
    reasons = set(record["quarantine"]["reasons"])
    transformations = set(record["transformations"])
    eligible = bool(record["release_eligibility"]["eligible"])

    expected_eligible = bool(case.get("expected_eligible"))
    if eligible != expected_eligible:
        blockers.append(f"eligible_expected_{expected_eligible}_got_{eligible}")

    for reason in case.get("expected_reasons_present") or []:
        if reason not in reasons:
            blockers.append(f"missing_reason:{reason}")
    for reason in case.get("expected_reasons_absent") or []:
        if reason in reasons:
            blockers.append(f"unexpected_reason:{reason}")

    for transform in case.get("expected_transformations_present") or []:
        if transform not in transformations:
            blockers.append(f"missing_transformation:{transform}")
    for transform in case.get("expected_transformations_absent") or []:
        if transform in transformations:
            blockers.append(f"unexpected_transformation:{transform}")

    text = str(record.get("text") or "")
    for expected in case.get("expected_text_contains") or []:
        if str(expected) not in text:
            blockers.append(f"missing_text:{expected}")

    hazards = record.get("hazards") or {}
    return {
        "id": str(case["id"]),
        "passed": not blockers,
        "blockers": blockers,
        "expected_eligible": expected_eligible,
        "eligible": eligible,
        "reasons": sorted(reasons),
        "transformations": sorted(transformations),
        "hazards": {
            "pii_detected": bool(hazards.get("pii_detected")),
            "secret_detected": bool(hazards.get("secret_detected")),
            "benchmark_contamination": bool(hazards.get("benchmark_contamination")),
            "poisoning_suspected": bool(hazards.get("poisoning_suspected")),
            "diagnostics": hazards.get("diagnostics") or {},
        },
    }


def build(fixtures_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    payload = load_json(fixtures_path)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Stage-0 hazard benchmark fixtures must contain a cases list.")
    rows = [_evaluate_case(case, index) for index, case in enumerate(cases)]
    blockers = [f"{row['id']}:{blocker}" for row in rows for blocker in row["blockers"]]
    reason_counts: Dict[str, int] = {}
    for row in rows:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "schema_version": "stage0-hazard-benchmark-report-v1",
        "status": "stage0_hazard_fixture_benchmark_passed" if not blockers else "stage0_hazard_fixture_benchmark_failed",
        "claim_boundary": payload.get("claim_boundary")
        or "Minimal labeled Stage-0 hazard fixture benchmark. This is not production detector validation.",
        "fixtures_path": str(fixtures_path),
        "summary": {
            "case_count": len(rows),
            "passed_count": sum(1 for row in rows if row["passed"]),
            "failed_count": sum(1 for row in rows if not row["passed"]),
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        },
        "cases": rows,
        "blockers": blockers,
        "remaining_evidence_gaps": [
            "fixture_benchmark_not_real_corpus_detector_validation",
            "license_policy_requires_source_specific_legal_metadata",
            "benchmark_contamination_requires_task_hash_or_canonical_benchmark_registry",
            "poisoning_detection_requires_adversarial_benchmark_expansion",
        ],
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Stage-0 Hazard Benchmark",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Summary",
        "",
        f"- Cases: `{report['summary']['case_count']}`",
        f"- Passed: `{report['summary']['passed_count']}`",
        f"- Failed: `{report['summary']['failed_count']}`",
        "",
        "## Cases",
        "",
        "| Case | Passed | Eligible | Reasons |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        lines.append(
            f"| `{row['id']}` | `{row['passed']}` | `{row['eligible']}` | `{', '.join(row['reasons']) or 'None'}` |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- `{blocker}`" for blocker in report["blockers"]] or ["- None"])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{gap}`" for gap in report["remaining_evidence_gaps"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-0 hazard benchmark report.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.fixtures, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"], "summary": report["summary"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
