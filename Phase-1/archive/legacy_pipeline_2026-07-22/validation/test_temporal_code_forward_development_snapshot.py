#!/usr/bin/env python3
"""Contract checks for forward development snapshots."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    plan = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_development_snapshot_plan.json"
    )
    candidates = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_development_candidates.json"
    )
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_forward_development_snapshot_report.json"
    )
    accumulation = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
    )
    capacity = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_forward_discovery_capacity_report.json"
    )
    broad = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
    )
    frame = set(plan["snapshot"]["repository_identities"])
    assert plan["status"] == "frozen_before_forward_development_task_metadata"
    assert plan["snapshot"]["window_start"] == "2026-06-15"
    assert plan["snapshot"]["available_through"] <= plan["snapshot"]["run_date"]
    assert not frame.intersection(broad["repositories"])
    assert plan["training_repository_overlap_count"] == 0
    assert plan["execution_outcomes_read"] is False
    assert candidates["status"] == "frozen_before_project_metadata_or_execution_outcomes"
    assert candidates["training_repository_overlap_count"] == 0
    assert all(row["assigned_split"] == "development" for row in candidates["candidates"])
    assert all(row["evaluation_authorized_pending_e2_and_quarantine"] is False for row in candidates["candidates"])
    assert candidates["execution_outcomes_read"] is False
    assert candidates["confirmatory_outcomes_read"] is False
    assert candidates["development_utility_may_start"] is False
    assert candidates["forbidden_fields_collected"] == []
    assert report["status"] == "forward_development_snapshot_complete_no_candidates"
    assert report["summary"]["metadata_candidate_count"] == 0
    assert report["summary"]["training_repository_overlap_count"] == 0
    assert report["decision"]["zero_candidates_is_valid_snapshot_evidence"] is True
    assert report["decision"]["retroactively_expand_same_snapshot_after_candidate_outcome"] is False
    assert report["decision"]["candidate_recipe_or_execution_may_start"] is False
    assert report["decision"]["development_utility_may_start"] is False
    assert report["decision"]["confirmatory_outcomes_read"] is False
    assert accumulation["status"] == "frozen_after_snapshot_001_metadata_and_before_any_later_snapshot_metadata"
    assert accumulation["contract"]["development_accumulation_amendment"]["eligibility_rule_changes"] == "none"
    assert accumulation["contract"]["development_accumulation_amendment"]["not_justified_by"] == (
        "snapshot_001_zero_candidate_outcome"
    )
    assert accumulation["accumulation_frame"]["existing_broad_repository_overlap_count"] == 0
    assert accumulation["accumulation_frame"]["benchmark_source_repository_overlap_count"] == 0
    assert accumulation["capacity_context"]["estimate_role"] == "planning_only"
    assert accumulation["capacity_context"]["frame_meets_point_estimate_candidate_capacity"] is True
    assert accumulation["capacity_context"]["frame_alone_guarantees_target"] is False
    assert accumulation["next_snapshot_task_metadata_read"] is False
    assert accumulation["development_utility_may_start"] is False
    assert capacity["status"] == "forward_repository_frame_meets_point_estimate_candidate_capacity"
    assert capacity["summary"]["combined_discovered_repository_count"] == 12067
    assert capacity["summary"]["frozen_fresh_repository_frame_count"] == 5000
    assert capacity["summary"]["point_estimate_candidate_capacity_met"] is True
    assert capacity["decision"]["structural_repository_frame_blocker_resolved"] is True
    assert capacity["decision"]["actual_task_distribution_blocker_resolved"] is False
    assert capacity["decision"]["frame_guarantees_task_target"] is False
    assert capacity["decision"]["development_utility_may_start"] is False
    assert capacity["task_metadata_read_from_expanded_frame"] is False
    print("[temporal-code-forward-development] outcome-free snapshot boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
