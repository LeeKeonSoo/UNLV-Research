#!/usr/bin/env python3
"""Report retrospective development acquisition progress and capacity."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_SCHEDULE = COLLECTION / "temporal_code_retrospective_development_schedule.json"
DEFAULT_LEDGER = COLLECTION / "temporal_code_retrospective_candidate_ledger.json"
DEFAULT_E2 = OUTPUT_DIR / "validation" / "temporal_code_retrospective_e2_batch_000_report.json"
DEFAULT_DISCOVERY = COLLECTION / "forward_development_repository_discovery_combined.json"
DEFAULT_BROAD = COLLECTION / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_retrospective_development_report.json"


def _ceil_division(target: int, rate: float) -> int | None:
    return math.ceil(target / rate) if rate > 0 else None


def build(
    schedule_path: Path,
    ledger_path: Path,
    e2_path: Path,
    discovery_path: Path,
    broad_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    schedule = load_json(schedule_path)
    ledger = load_json(ledger_path)
    e2 = load_json(e2_path)
    discovery = load_json(discovery_path)
    broad = load_json(broad_path)
    repository_count = int(schedule["summary"]["repository_count"])
    candidate_count = int(ledger["summary"]["candidate_count"])
    target = int(ledger["summary"]["development_target_task_count"])
    batch_summary = e2["summary"]
    attempted = int(batch_summary["execution_candidate_count"])
    valid = int(batch_summary["task_valid_e2_count"])
    candidate_rate = candidate_count / repository_count if repository_count else 0.0
    e2_rate = valid / attempted if attempted else 0.0
    candidates_needed = _ceil_division(target, e2_rate)
    candidate_gap = max(0, int(candidates_needed or 0) - candidate_count)
    additional_repositories_needed = _ceil_division(candidate_gap, candidate_rate)
    combined_ids = set(discovery["candidates"])
    initial_ids = {identity for shard in schedule["shards"] for identity in shard["repository_identities"]}
    broad_ids = set(broad["repositories"])
    combined_count = len(combined_ids)
    remaining_repositories = len(combined_ids - initial_ids - broad_ids)
    excluded_training_repositories = len((combined_ids - initial_ids).intersection(broad_ids))
    projected_current_e2 = math.floor(candidate_count * e2_rate)
    projected_all_frame_candidates = candidate_count + math.floor(remaining_repositories * candidate_rate)
    projected_all_frame_e2 = math.floor(projected_all_frame_candidates * e2_rate)
    failures = Counter(row.get("failure_stage") or "task_valid_e2" for row in e2["decisions"])
    report = {
        "schema_version": "temporal-code-retrospective-development-report-v1",
        "status": "retrospective_development_feasible_expansion_required_before_utility",
        "source_sha256": {
            str(schedule_path): sha256_file(schedule_path),
            str(ledger_path): sha256_file(ledger_path),
            str(e2_path): sha256_file(e2_path),
            str(discovery_path): sha256_file(discovery_path),
            str(broad_path): sha256_file(broad_path),
        },
        "observed": {
            "repositories_scanned": repository_count,
            "immutable_metadata_shards": int(schedule["summary"]["shard_count"]),
            "strict_metadata_candidates": candidate_count,
            "candidate_per_repository_rate": candidate_rate,
            "first_e2_batch_execution_attempts": attempted,
            "first_e2_batch_task_valid_count": valid,
            "first_e2_batch_task_valid_rate": e2_rate,
            "first_e2_batch_failure_stage_counts": dict(sorted(failures.items())),
            "training_repository_overlap_count": int(schedule["summary"]["training_repository_overlap_count"]),
        },
        "planning_estimate_only": {
            "development_valid_e2_target": target,
            "metadata_candidates_needed_at_first_batch_e2_rate": candidates_needed,
            "current_candidate_gap_to_point_estimate": candidate_gap,
            "projected_valid_e2_from_current_candidates": projected_current_e2,
            "combined_discovery_repository_count": combined_count,
            "remaining_unscanned_repository_count": remaining_repositories,
            "remaining_frame_training_repository_exclusion_count": excluded_training_repositories,
            "additional_repositories_needed_at_observed_rates": additional_repositories_needed,
            "projected_candidates_after_scanning_entire_combined_frame": projected_all_frame_candidates,
            "projected_valid_e2_after_scanning_entire_combined_frame": projected_all_frame_e2,
            "entire_combined_frame_meets_point_estimate": projected_all_frame_e2 >= target,
            "inferential_capacity_claim_allowed": False,
        },
        "decision": {
            "retrospective_acquisition_produced_task_valid_e2": valid > 0,
            "actual_task_distribution_blocker_resolved": valid >= target,
            "same_rules_full_remaining_frame_expansion_justified": (
                valid > 0 and projected_current_e2 < target and remaining_repositories > 0
            ),
            "task_validity_rule_may_be_weakened": False,
            "one_task_per_repository_may_be_weakened": False,
            "development_utility_may_start": valid >= target,
            "confirmatory_outcomes_read": False,
            "next_action": (
                "freeze and scan the remaining disjoint retrospective repository frame under unchanged rules"
            ),
        },
        "utility_scope": ledger["utility_scope"],
        "claim_boundary": (
            "Retrospective development acquisition and E2 productivity only; no Utility, selector, "
            "curation-benefit, or release claim."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrospective development acquisition report.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--e2", type=Path, default=DEFAULT_E2)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.schedule, args.ledger, args.e2, args.discovery, args.broad, args.output)
    print({"status": report["status"], "observed": report["observed"], "planning": report["planning_estimate_only"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
