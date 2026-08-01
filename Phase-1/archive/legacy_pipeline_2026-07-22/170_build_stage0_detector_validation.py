#!/usr/bin/env python3
"""Build labeled Stage-0 detector validation precheck report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.normalize import process_candidate


DEFAULT_FIXTURES = Path("validation") / "fixtures" / "stage0_detector_validation_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage0_detector_validation_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage0_detector_validation_report.md"
AXES = ("pii_detected", "secret_detected", "benchmark_contamination", "poisoning_suspected", "rights_allowed")
DEFAULT_REMAINING_GAPS = [
    "fixture_distribution_is_small_and_project_defined",
    "requires_external_labeled_detector_benchmark_before_production_claim",
    "license_policy_still_requires_source_specific_legal_metadata",
    "benchmark_contamination_requires_canonical_task_hash_registry_for_full_coverage",
    "poisoning_detection_requires_larger_adversarial_benchmark",
]


def _raw_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(case["id"]),
        "text": str(case.get("text") or ""),
        "pii_context": str(case.get("pii_context") or "general"),
        "source_name": "stage0-detector-validation-fixture",
        "source_uri": f"fixture://stage0-detector-validation/{case['id']}",
        "collected_at": "2026-06-23T00:00:00Z",
        "language": {"code": "en", "confidence": 1.0},
        "rights": case.get("rights") if isinstance(case.get("rights"), dict) else {"status": "unknown", "license": None},
    }


def _actual_axis(record: Dict[str, Any], axis: str) -> bool:
    if axis == "rights_allowed":
        return (record.get("rights") or {}).get("status") == "allowed"
    return bool((record.get("hazards") or {}).get(axis))


def _empty_confusion() -> Dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0}


def _rates(values: Dict[str, int]) -> Dict[str, float]:
    precision_den = values["tp"] + values["fp"]
    recall_den = values["tp"] + values["fn"]
    return {
        "precision": round(values["tp"] / precision_den, 6) if precision_den else 1.0,
        "recall": round(values["tp"] / recall_den, 6) if recall_den else 1.0,
        "false_positive_count": int(values["fp"]),
        "false_negative_count": int(values["fn"]),
    }


def _evaluate_case(case: Dict[str, Any], index: int) -> Dict[str, Any]:
    record = process_candidate(_raw_case(case), index=index)
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    axes = {}
    blockers: List[str] = []
    for axis in AXES:
        expected_value = bool(expected.get(axis))
        actual_value = _actual_axis(record, axis)
        axes[axis] = {"expected": expected_value, "actual": actual_value}
        if expected_value != actual_value:
            blockers.append(f"{axis}_expected_{expected_value}_got_{actual_value}")
    expected_eligible = bool(expected.get("eligible"))
    actual_eligible = bool((record.get("release_eligibility") or {}).get("eligible"))
    if expected_eligible != actual_eligible:
        blockers.append(f"eligible_expected_{expected_eligible}_got_{actual_eligible}")
    return {
        "id": str(case["id"]),
        "passed": not blockers,
        "blockers": blockers,
        "axes": axes,
        "expected_eligible": expected_eligible,
        "actual_eligible": actual_eligible,
        "quarantine_reasons": (record.get("quarantine") or {}).get("reasons") or [],
        "diagnostics": (record.get("hazards") or {}).get("diagnostics") or {},
    }


def build(fixtures_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    payload = load_json(fixtures_path)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Stage-0 detector validation fixtures must contain a cases list.")
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    min_recall = float(thresholds.get("minimum_recall") or 1.0)
    min_precision = float(thresholds.get("minimum_precision") or 0.8)

    rows = [_evaluate_case(case, index) for index, case in enumerate(cases)]
    confusion = {axis: _empty_confusion() for axis in AXES}
    examples = {axis: {"false_positives": [], "false_negatives": []} for axis in AXES}
    for row in rows:
        for axis, values in row["axes"].items():
            expected = bool(values["expected"])
            actual = bool(values["actual"])
            if expected and actual:
                confusion[axis]["tp"] += 1
            elif expected and not actual:
                confusion[axis]["fn"] += 1
                examples[axis]["false_negatives"].append(row["id"])
            elif not expected and actual:
                confusion[axis]["fp"] += 1
                examples[axis]["false_positives"].append(row["id"])
            else:
                confusion[axis]["tn"] += 1

    metrics = {
        axis: {**values, **_rates(values), **examples[axis]}
        for axis, values in confusion.items()
    }
    blockers = [f"{row['id']}:{blocker}" for row in rows for blocker in row["blockers"]]
    for axis, values in metrics.items():
        if values["recall"] < min_recall:
            blockers.append(f"{axis}_recall_below_threshold:{values['recall']}")
        if values["precision"] < min_precision:
            blockers.append(f"{axis}_precision_below_threshold:{values['precision']}")

    pass_status = str(
        payload.get("pass_status")
        or "stage0_detector_validation_precheck_passed_with_scope_caveats"
    )
    fail_status = str(payload.get("fail_status") or "stage0_detector_validation_precheck_failed")
    remaining_gaps = payload.get("remaining_evidence_gaps")
    if not isinstance(remaining_gaps, list):
        remaining_gaps = DEFAULT_REMAINING_GAPS

    report = {
        "schema_version": "stage0-detector-validation-report-v1",
        "status": pass_status if not blockers else fail_status,
        "benchmark_scope": str(payload.get("benchmark_scope") or "project_defined_precheck"),
        "heldout_from": payload.get("heldout_from"),
        "source_policy": payload.get("source_policy"),
        "claim_boundary": payload.get("claim_boundary")
        or "Labeled detector-validation precheck, not external production detector certification.",
        "fixtures_path": str(fixtures_path),
        "thresholds": {"minimum_recall": min_recall, "minimum_precision": min_precision},
        "summary": {
            "case_count": len(rows),
            "passed_count": sum(1 for row in rows if row["passed"]),
            "failed_count": sum(1 for row in rows if not row["passed"]),
            "axis_count": len(AXES),
        },
        "axis_metrics": metrics,
        "cases": rows,
        "blockers": blockers,
        "remaining_evidence_gaps": [str(gap) for gap in remaining_gaps],
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Stage-0 Detector Validation Precheck",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Summary",
        "",
        f"- Benchmark scope: `{report['benchmark_scope']}`",
        f"- Cases: `{report['summary']['case_count']}`",
        f"- Passed: `{report['summary']['passed_count']}`",
        f"- Failed: `{report['summary']['failed_count']}`",
        "",
        "## Axis Metrics",
        "",
        "| Axis | TP | FP | TN | FN | Precision | Recall |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for axis, values in report["axis_metrics"].items():
        lines.append(
            f"| `{axis}` | `{values['tp']}` | `{values['fp']}` | `{values['tn']}` | `{values['fn']}` | "
            f"`{values['precision']}` | `{values['recall']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- `{blocker}`" for blocker in report["blockers"]] or ["- None"])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{gap}`" for gap in report["remaining_evidence_gaps"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-0 detector validation precheck report.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.fixtures, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"], "summary": report["summary"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
