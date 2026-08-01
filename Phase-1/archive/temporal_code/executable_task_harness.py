#!/usr/bin/env python3
"""Freeze the independent executable-task harness acquisition contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_executable_task_harness_v1.json"
DEFAULT_EXECUTION_SUPPORT = OUTPUT_DIR / "validation" / "temporal_code_execution_support_report.json"
DEFAULT_FRESH_EXPANSION = OUTPUT_DIR / "validation" / "temporal_code_development_fresh_expansion_report.json"
DEFAULT_SOURCE_PROFILE = OUTPUT_DIR / "temporal_code_collection" / "swebench_harness_metadata_profile.json"
DEFAULT_EVALPLUS_PREVALIDATION = OUTPUT_DIR / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
DEFAULT_EVALPLUS_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_RETENTION_GUARDRAILS = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
)
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_executable_task_harness_plan.json"


def freeze(
    contract_path: Path,
    execution_support_path: Path,
    fresh_expansion_path: Path,
    source_profile_path: Path,
    evalplus_prevalidation_path: Path,
    evalplus_split_path: Path,
    retention_guardrails_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    execution_support = load_json(execution_support_path)
    fresh_expansion = load_json(fresh_expansion_path)
    source_profile = load_json(source_profile_path) if source_profile_path.exists() else {}
    evalplus_prevalidation = (
        load_json(evalplus_prevalidation_path) if evalplus_prevalidation_path.exists() else {}
    )
    evalplus_split = load_json(evalplus_split_path) if evalplus_split_path.exists() else {}
    retention_guardrails = (
        load_json(retention_guardrails_path) if retention_guardrails_path.exists() else {}
    )
    blockers = [
        "primary_executable_aggregate_not_frozen",
        "primary_temporal_development_and_confirmatory_e2_task_pools_not_acquired",
    ]
    if retention_guardrails.get("status") != "frozen_before_development_model_outcomes":
        blockers.append("general_retention_non_inferiority_guardrails_not_frozen")
    precision = source_profile.get("precision_analysis") or {}
    e2_analysis = source_profile.get("e2_analysis") or {}
    if source_profile and not precision.get("eligible_count_meets_required_task_count"):
        blockers.append("swebench_verified_candidate_count_below_frozen_precision_requirement")
    if source_profile and int(e2_analysis.get("e2_verified_task_count") or 0) == 0:
        blockers.append("swebench_verified_candidates_not_e2_prevalidated")
    if evalplus_prevalidation.get("status") == "platform_runtime_blocked_before_semantic_controls":
        blockers.append("evalplus_windows_native_runtime_and_isolation_blocked")
    status = (
        "frozen_contract_source_profiled_e2_acquisition_blocked"
        if source_profile
        else "frozen_contract_pending_task_acquisition"
    )
    report = {
        "schema_version": "temporal-code-executable-task-harness-plan-v1",
        "status": status,
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(execution_support_path): sha256_file(execution_support_path),
            str(fresh_expansion_path): sha256_file(fresh_expansion_path),
            **({str(source_profile_path): sha256_file(source_profile_path)} if source_profile else {}),
            **(
                {str(evalplus_prevalidation_path): sha256_file(evalplus_prevalidation_path)}
                if evalplus_prevalidation
                else {}
            ),
            **({str(evalplus_split_path): sha256_file(evalplus_split_path)} if evalplus_split else {}),
            **(
                {str(retention_guardrails_path): sha256_file(retention_guardrails_path)}
                if retention_guardrails
                else {}
            ),
        },
        "current_evidence": {
            "execution_support_status": execution_support["status"],
            "audited_bundle_count": execution_support["summary"]["bundle_count"],
            "executable_stage_c_eligible_bundle_count": execution_support["summary"][
                "executable_stage_c_eligible_count"
            ],
            "fresh_transfer_status": fresh_expansion["status"],
            "source_profile_status": source_profile.get("status"),
            "source_split_summary": source_profile.get("split_summary") or {},
            "source_precision_analysis": precision,
            "source_e2_analysis": e2_analysis,
            "evalplus_guardrail_status": evalplus_prevalidation.get("status"),
            "evalplus_guardrail_decision": evalplus_prevalidation.get("decision") or {},
            "evalplus_guardrail_split_status": evalplus_split.get("status"),
            "evalplus_guardrail_split_summary": evalplus_split.get("summary") or {},
            "retention_guardrail_status": retention_guardrails.get("status"),
            "development_utility_may_start": False,
        },
        "source_assessment": {
            "swebench_verified_role": "secondary_repository_level_guardrail_candidate",
            "suitable_as_sole_primary_executable_aggregate": False if source_profile else None,
            "reason": (
                "The frozen repository/time-disjoint candidate pool is below the frozen precision requirement "
                "and no candidate is locally E2-prevalidated."
                if source_profile
                else "Source metadata acquisition is pending."
            ),
        },
        "entry_blockers": blockers,
        "next_actions": [
            "acquire development and untouched confirmatory E2 tasks independently from candidate training-corpus collection",
            "acquire a larger task-class-specific E2 function-level harness for the primary aggregate",
            "run the frozen EvalPlus prevalidation in WSL2/Linux with an isolated execution backend",
            "retain SWE-bench Verified as a secondary repository-level guardrail after E2 prevalidation",
            "freeze the primary executable aggregate and retention non-inferiority guardrails",
        ],
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze executable task harness acquisition contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--execution-support", type=Path, default=DEFAULT_EXECUTION_SUPPORT)
    parser.add_argument("--fresh-expansion", type=Path, default=DEFAULT_FRESH_EXPANSION)
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_SOURCE_PROFILE)
    parser.add_argument("--evalplus-prevalidation", type=Path, default=DEFAULT_EVALPLUS_PREVALIDATION)
    parser.add_argument("--evalplus-split", type=Path, default=DEFAULT_EVALPLUS_SPLIT)
    parser.add_argument("--retention-guardrails", type=Path, default=DEFAULT_RETENTION_GUARDRAILS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(
        args.contract,
        args.execution_support,
        args.fresh_expansion,
        args.source_profile,
        args.evalplus_prevalidation,
        args.evalplus_split,
        args.retention_guardrails,
        args.output,
    )
    print({"status": report["status"], "entry_blockers": report["entry_blockers"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
