#!/usr/bin/env python3
"""Build primary temporal executable-source feasibility decision."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_primary_executable_source_assessment_v1.json"
DEFAULT_HARNESS = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_executable_task_harness_plan.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_primary_executable_source_assessment.json"


def build(contract_path: Path, harness_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    harness = load_json(harness_path)
    required = int(contract["required_primary_contract"]["required_task_count"])
    sources = contract["source_snapshots"]
    current_primary_e2 = int(
        sources["project_created_recent_repository_tasks"]["current_development_e2_tasks"]
    ) + int(sources["project_created_recent_repository_tasks"]["current_confirmatory_e2_tasks"])
    report = {
        "schema_version": "temporal-code-primary-executable-source-assessment-report-v1",
        "status": "primary_temporal_executable_distribution_not_currently_acquirable_from_frozen_sources",
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(harness_path): sha256_file(harness_path),
        },
        "summary": {
            "required_primary_task_count": required,
            "current_primary_temporal_e2_task_count": current_primary_e2,
            "task_count_gap": required - current_primary_e2,
            "evalplus_e2_guardrail_frozen": (
                harness["current_evidence"].get("evalplus_guardrail_split_status")
                == "frozen_e2_guardrail_split_before_model_outcomes"
            ),
            "current_public_source_meets_primary_contract": False,
        },
        "decision": {
            "development_utility_may_start": False,
            "action": "abstain_and_continue_forward_task_acquisition",
            "reason": (
                "Current project-created temporal E2 tasks are far below the frozen precision requirement; "
                "SWE-bench Verified and EvalPlus are secondary guardrails, and the current LiveCodeBench "
                "snapshot neither meets the frozen task count nor demonstrates untouched post-training-window "
                "confirmatory coverage."
            ),
            "retroactive_contract_weakening_allowed": False,
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build primary temporal executable source assessment.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--harness", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.contract, args.harness, args.output)
    print({"status": report["status"], "summary": report["summary"], "decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
