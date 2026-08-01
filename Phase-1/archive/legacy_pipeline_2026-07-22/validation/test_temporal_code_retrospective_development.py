#!/usr/bin/env python3
"""Contract checks for retrospective development acquisition."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    validation = PROJECT_DIR / "outputs" / "validation"
    collection = PROJECT_DIR / "outputs" / "temporal_code_collection"
    report = load_json(validation / "temporal_code_retrospective_development_report.json")
    expansion = load_json(collection / "temporal_code_retrospective_expansion_schedule.json")
    ledger = load_json(collection / "temporal_code_retrospective_combined_candidate_ledger.json")
    status = load_json(validation / "temporal_code_retrospective_operations_status.json")
    capacity = load_json(validation / "temporal_code_retrospective_e2_capacity_audit.json")
    observed = report["observed"]
    planning = report["planning_estimate_only"]
    decision = report["decision"]
    assert report["status"] == "retrospective_development_feasible_expansion_required_before_utility"
    assert observed["repositories_scanned"] == 5000
    assert observed["strict_metadata_candidates"] == 1666
    assert observed["first_e2_batch_execution_attempts"] == 25
    assert observed["first_e2_batch_task_valid_count"] == 4
    assert observed["training_repository_overlap_count"] == 0
    assert planning["development_valid_e2_target"] == 542
    assert planning["remaining_unscanned_repository_count"] == 6822
    assert planning["remaining_frame_training_repository_exclusion_count"] == 245
    assert planning["entire_combined_frame_meets_point_estimate"] is True
    assert planning["inferential_capacity_claim_allowed"] is False
    assert decision["actual_task_distribution_blocker_resolved"] is False
    assert decision["same_rules_full_remaining_frame_expansion_justified"] is True
    assert decision["task_validity_rule_may_be_weakened"] is False
    assert decision["development_utility_may_start"] is False
    assert decision["confirmatory_outcomes_read"] is False
    assert expansion["status"] == "frozen_after_first_e2_batch_and_before_remaining_repository_task_metadata"
    assert expansion["summary"]["combined_repository_count"] == 12067
    assert expansion["summary"]["initial_repository_count"] == 5000
    assert expansion["summary"]["remaining_repository_count"] == 6822
    assert expansion["summary"]["shard_count"] == 35
    assert expansion["summary"]["initial_repository_overlap_count"] == 0
    assert expansion["summary"]["training_repository_overlap_count"] == 0
    assert expansion["adaptation_contract"]["eligibility_rule_changes"] == "none"
    assert expansion["adaptation_contract"]["task_validity_rule_changes"] == "none"
    assert expansion["confirmatory_outcomes_read"] is False
    assert expansion["development_utility_may_start"] is False
    assert ledger["status"] == "combined_retrospective_candidate_ledger_frozen_before_recipe_or_execution"
    assert ledger["summary"]["scheduled_repository_count"] == 11822
    assert ledger["summary"]["snapshot_count"] == 60
    assert ledger["summary"]["expected_snapshot_count"] == 60
    assert ledger["summary"]["metadata_collection_complete"] is True
    assert ledger["summary"]["candidate_count"] == 3847
    assert status["status"] == "retrospective_collection_ready_for_remaining_e2_batches"
    assert status["summary"]["e2_execution_attempt_count"] == 825
    assert status["summary"]["task_valid_e2_count"] == 167
    assert status["summary"]["valid_e2_gap"] == 375
    assert status["gates"]["metadata_collection_complete"] is True
    assert status["gates"]["development_utility_may_start"] is False
    assert status["gates"]["confirmatory_outcomes_read"] is False
    assert capacity["status"] == "retrospective_strict_e2_execution_should_continue"
    assert capacity["observed"]["execution_attempt_count"] == 825
    assert capacity["observed"]["task_valid_e2_count"] == 167
    assert capacity["decision"]["strict_e2_execution_may_continue"] is True
    assert capacity["decision"]["development_utility_may_start"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-retrospective] unchanged-rule expansion frozen: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
