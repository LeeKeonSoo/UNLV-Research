#!/usr/bin/env python3
"""Build orthogonal training-content and execution-support tier evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_CONTRACT = Path("configs") / "temporal_code_execution_support_tiers_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_execution_support_report.json"
COHORTS = {
    "path_stratified_primary": (
        COLLECTION / "path_stratified_tranche_bundles" / "path_stratified_tranche_bundle_audit_report.json"
    ),
    "confirmatory_expansion": (
        COLLECTION
        / "confirmatory_execution_expansion_bundles"
        / "confirmatory_execution_expansion_audit_report.json"
    ),
    "development_code_and_test_expansion": (
        COLLECTION
        / "development_execution_expansion_bundles"
        / "development_execution_expansion_audit_report.json"
    ),
    "development_fresh_test_or_code_expansion": (
        COLLECTION
        / "development_fresh_expansion_bundles"
        / "development_fresh_expansion_audit_report.json"
    ),
}


def _cohort(path: Path) -> Dict[str, Any]:
    audit = load_json(path)
    rows = audit["decisions"]
    matrix = {"C0/E0": 0, "C0/E1": 0, "C0/E2": 0, "C1/E0": 0, "C1/E1": 0, "C1/E2": 0}
    for row in rows:
        content = "C1" if row.get("collection_gate_pass") is True else "C0"
        if row.get("executable_evaluation_gate_pass") is True:
            execution = "E2"
        elif "test_command_not_verified" in set(row.get("executable_evaluation_blockers") or []):
            execution = "E1"
        else:
            execution = "E0"
        matrix[f"{content}/{execution}"] += 1
    return {
        "bundle_count": len(rows),
        "training_content_eligible_count": sum(row.get("collection_gate_pass") is True for row in rows),
        "executable_stage_c_eligible_count": sum(
            row.get("executable_evaluation_gate_pass") is True for row in rows
        ),
        "tier_matrix": matrix,
    }


def build(contract_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    cohorts = {name: _cohort(path) for name, path in COHORTS.items()}
    totals = {
        "bundle_count": sum(row["bundle_count"] for row in cohorts.values()),
        "training_content_eligible_count": sum(
            row["training_content_eligible_count"] for row in cohorts.values()
        ),
        "executable_stage_c_eligible_count": sum(
            row["executable_stage_c_eligible_count"] for row in cohorts.values()
        ),
    }
    report = {
        "schema_version": "temporal-code-execution-support-report-v1",
        "status": "orthogonal_content_and_execution_tiers_operational",
        "contract": contract,
        "cohorts": cohorts,
        "summary": totals,
        "decision": {
            "training_content_may_be_preserved_without_executable_support": True,
            "executable_stage_c_requires_e2": True,
            "execution_tier_may_enter_stage_b": False,
            "development_utility_may_start": False,
            "reason": "Only one development E2 bundle exists; a representative executable task distribution is absent.",
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal-code execution-support report.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.contract, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
