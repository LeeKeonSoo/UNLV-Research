#!/usr/bin/env python3
"""Build the fresh repository-disjoint development expansion decision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_PLAN = COLLECTION / "temporal_code_development_fresh_expansion_plan.json"
DEFAULT_FETCH = COLLECTION / "development_fresh_expansion_bundles" / "smoke_fetch_report.json"
DEFAULT_AUDIT = COLLECTION / "development_fresh_expansion_bundles" / "development_fresh_expansion_audit_report.json"
DEFAULT_GENERIC = COLLECTION / "development_fresh_generic_verification.json"
DEFAULT_NATIVE = COLLECTION / "development_fresh_native_verification.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_development_fresh_expansion_report.json"


def _build_pass_count(report: Dict[str, Any]) -> int:
    return sum(
        result["build"]["exit_code"] == 0
        for row in report["decisions"]
        for result in row.get("commit_results") or []
    )


def _test_pass_count(report: Dict[str, Any]) -> int:
    return sum(
        result["test"]["exit_code"] == 0
        for row in report["decisions"]
        for result in row.get("commit_results") or []
    )


def build(
    plan_path: Path,
    fetch_path: Path,
    audit_path: Path,
    generic_path: Path,
    native_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    fetch = load_json(fetch_path)
    audit = load_json(audit_path)
    generic = load_json(generic_path)
    native = load_json(native_path)
    report = {
        "schema_version": "temporal-code-development-fresh-expansion-report-v1",
        "status": "raw_repository_execution_support_insufficient",
        "summary": {
            "fresh_repository_count": int(plan["summary"]["repository_count"]),
            "fresh_path_stratum_counts": plan["summary"]["path_stratum_counts"],
            "fetched_bundles": int(fetch["summary"]["bundle_count"]),
            "fetched_file_records": int(fetch["summary"]["file_record_count"]),
            "collection_gate_pass_bundles": int(audit["summary"]["collection_gate_pass_count"]),
            "generic_execution_candidates": int(generic["summary"]["bundle_count"]),
            "generic_build_pass_commits": _build_pass_count(generic),
            "generic_test_pass_commits": _test_pass_count(generic),
            "generic_verified_bundles": int(generic["summary"]["verified_bundle_count"]),
            "native_build_pass_commits": _build_pass_count(native),
            "native_test_pass_commits": _test_pass_count(native),
            "native_verified_bundles": int(native["summary"]["verified_bundle_count"]),
            "total_verified_development_bundles": 1,
        },
        "decision": {
            "development_utility_may_start": False,
            "broaden_raw_repository_discovery_for_execution_recovery": False,
            "reason": (
                "The unchanged metadata-derived native recipe recovered no executable bundle on a fresh "
                "repository-disjoint development expansion. Broadening raw-repository discovery alone is "
                "unlikely to solve the execution-support bottleneck."
            ),
            "required_architecture_change": (
                "Separate candidate-corpus curation from executable-task acquisition. Stage C requires a "
                "prevalidated executable-task harness or an explicit repository execution-support tier."
            ),
            "allowed_next_actions": [
                "freeze explicit repository execution-support tiers and define which tiers may enter executable Stage C",
                "freeze an independently sourced prevalidated temporal executable-task harness",
                "use external code benchmarks and retention tasks as Stage-C guardrails while preserving temporal executable tasks as a separate evidence stream",
            ],
            "forbidden_reactions": [
                "infer Stage-B, curation, or Utility failure from repository execution-support failure",
                "tune Stage B from execution or Utility outcomes",
                "add repository-specific execution exceptions from these development outcomes",
                "inspect confirmatory outcomes",
            ],
        },
        "interpretation": (
            "Raw collected repositories remain valid candidate training data after content gates, but they do not "
            "automatically constitute reproducible executable evaluation tasks. These are separate acquisition capabilities."
        ),
        "confirmatory_outcomes_read": False,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "The raw-repository execution-support bottleneck is demonstrated across two repository-disjoint "
            "development expansions. No Utility, curation-benefit, curation-failure, or release claim is established."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fresh development expansion report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--fetch", type=Path, default=DEFAULT_FETCH)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--generic", type=Path, default=DEFAULT_GENERIC)
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.plan, args.fetch, args.audit, args.generic, args.native, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
