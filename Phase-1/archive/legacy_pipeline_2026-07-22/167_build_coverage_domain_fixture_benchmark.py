#!/usr/bin/env python3
"""Build a coverage/domain metadata fixture benchmark report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from policy.subsets import _bucket_support_pass, _distribution_bucket_support, _domain_bucket_from_row, _source_bucket_support_scope


DEFAULT_FIXTURES = Path("validation") / "fixtures" / "coverage_domain_fixture_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "coverage_domain_fixture_benchmark_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "coverage_domain_fixture_benchmark_report.md"


def _bucket_counts(rows: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        counts[_domain_bucket_from_row(json.dumps(metadata, ensure_ascii=False), row.get("source"), row.get("input_source"))] += 1
    return counts


def _evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    original_counts = _bucket_counts(case.get("original") or [])
    selected_counts = _bucket_counts(case.get("selected") or [])
    scope = _source_bucket_support_scope(original_counts)
    support = _distribution_bucket_support(
        selected_counts,
        original_counts,
        support_scope=scope,
        support_label="source_or_domain_bucket",
    )
    threshold_pass = _bucket_support_pass(
        support,
        min_distribution_similarity=float(case.get("min_distribution_similarity") or 0.0),
        min_retained_bucket_ratio=float(case.get("min_retained_bucket_ratio") or 0.0),
    )
    blockers: List[str] = []
    expected_scope = str(case.get("expected_scope") or "")
    if scope != expected_scope:
        blockers.append(f"scope_expected_{expected_scope}_got_{scope}")
    expected_pass = bool(case.get("expected_pass"))
    if threshold_pass != expected_pass:
        blockers.append(f"pass_expected_{expected_pass}_got_{threshold_pass}")
    expected_buckets = set(str(value) for value in case.get("expected_original_buckets") or [])
    if expected_buckets and set(original_counts) != expected_buckets:
        blockers.append(f"original_buckets_expected_{sorted(expected_buckets)}_got_{sorted(original_counts)}")
    true_domain_claim_allowed = scope == "explicit_domain_metadata"
    if bool(case.get("must_not_claim_true_domain")) and true_domain_claim_allowed:
        blockers.append("true_domain_claim_unexpectedly_allowed")
    return {
        "id": str(case["id"]),
        "passed": not blockers,
        "blockers": blockers,
        "original_counts": dict(sorted(original_counts.items())),
        "selected_counts": dict(sorted(selected_counts.items())),
        "support": support,
        "threshold_pass": bool(threshold_pass),
        "true_domain_claim_allowed": bool(true_domain_claim_allowed),
        "expected_scope": expected_scope,
        "expected_pass": expected_pass,
    }


def build(fixtures_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    payload = load_json(fixtures_path)
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Coverage/domain fixture file must contain a cases list.")
    rows = [_evaluate_case(case) for case in cases]
    blockers = [f"{row['id']}:{blocker}" for row in rows for blocker in row["blockers"]]
    scope_counts = Counter(str((row.get("support") or {}).get("support_scope") or "unknown") for row in rows)
    report = {
        "schema_version": "coverage-domain-fixture-benchmark-report-v1",
        "status": "coverage_domain_fixture_benchmark_passed" if not blockers else "coverage_domain_fixture_benchmark_failed",
        "claim_boundary": payload.get("claim_boundary")
        or "Observable domain/source coverage benchmark. True domain coverage requires explicit metadata.",
        "fixtures_path": str(fixtures_path),
        "summary": {
            "case_count": len(rows),
            "passed_count": sum(1 for row in rows if row["passed"]),
            "failed_count": sum(1 for row in rows if not row["passed"]),
            "support_scope_counts": dict(sorted(scope_counts.items())),
        },
        "cases": rows,
        "blockers": blockers,
        "remaining_evidence_gaps": [
            "fixture_benchmark_not_real_corpus_metadata_validation",
            "semantic_domain_labels_still_require_dataset_specific_taxonomy",
            "source_bucket_fallback_must_not_be_reported_as_true_domain_coverage",
        ],
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Coverage Domain Fixture Benchmark",
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
        f"- Support scopes: `{report['summary']['support_scope_counts']}`",
        "",
        "## Cases",
        "",
        "| Case | Passed | Scope | Threshold Pass | True Domain Claim Allowed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["cases"]:
        support = row.get("support") or {}
        lines.append(
            f"| `{row['id']}` | `{row['passed']}` | `{support.get('support_scope')}` | `{row['threshold_pass']}` | `{row['true_domain_claim_allowed']}` |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- `{blocker}`" for blocker in report["blockers"]] or ["- None"])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{gap}`" for gap in report["remaining_evidence_gaps"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Coverage/domain fixture benchmark report.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.fixtures, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"], "summary": report["summary"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
