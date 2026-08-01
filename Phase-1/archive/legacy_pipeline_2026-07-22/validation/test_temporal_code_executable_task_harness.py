#!/usr/bin/env python3
"""Contract checks for independent executable-task harness acquisition."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    plan = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_executable_task_harness_plan.json"
    )
    contract = plan["contract"]
    assert plan["status"] == "frozen_contract_source_profiled_e2_acquisition_blocked"
    assert contract["task_role"] == "evaluation_only_never_training"
    assert contract["eligibility"]["execution_support_tier"] == "E2"
    assert contract["sample_size_rule"]["fixed_arbitrary_minimum_forbidden"] is True
    assert contract["sample_size_rule"]["practical_effect_margin_absolute"] == 0.05
    assert contract["sample_size_rule"]["training_seed_count"] == 5
    assert set(contract["task_class_e2_contracts"]) == {"repository_patch", "function_generation", "other"}
    assert "using task outcomes in Stage B" in contract["forbidden_uses"]
    assert "using different Stage-A baselines for sensitivity arms" in contract["forbidden_uses"]
    assert plan["current_evidence"]["development_utility_may_start"] is False
    assert plan["current_evidence"]["source_precision_analysis"]["required_task_count"] == 1083
    assert plan["current_evidence"]["source_precision_analysis"]["eligible_count_meets_required_task_count"] is False
    assert plan["current_evidence"]["source_e2_analysis"]["e2_verified_task_count"] == 0
    assert plan["current_evidence"]["evalplus_guardrail_status"] == "e2_prevalidated"
    assert plan["current_evidence"]["evalplus_guardrail_split_status"] == (
        "frozen_e2_guardrail_split_before_model_outcomes"
    )
    assert plan["current_evidence"]["evalplus_guardrail_split_summary"]["task_count"] == 542
    assert plan["current_evidence"]["retention_guardrail_status"] == "frozen_before_development_model_outcomes"
    assert "evalplus_windows_native_runtime_and_isolation_blocked" not in plan["entry_blockers"]
    assert "primary_temporal_development_and_confirmatory_e2_task_pools_not_acquired" in plan[
        "entry_blockers"
    ]
    assert "general_retention_non_inferiority_guardrails_not_frozen" not in plan["entry_blockers"]
    assert plan["source_assessment"]["suitable_as_sole_primary_executable_aggregate"] is False
    assert plan["confirmatory_outcomes_read"] is False
    assert plan["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-executable-harness] independent evaluation-only acquisition contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
