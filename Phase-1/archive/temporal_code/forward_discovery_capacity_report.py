#!/usr/bin/env python3
"""Report the frozen expanded repository-discovery capacity result."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_COMBINED = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_combined.json"
DEFAULT_ACCUMULATION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
DEFAULT_PRODUCTIVITY = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_discovery_capacity_report.json"


def build(combined_path: Path, accumulation_path: Path, productivity_path: Path, output_path: Path) -> Dict[str, Any]:
    combined = load_json(combined_path)
    accumulation = load_json(accumulation_path)
    productivity = load_json(productivity_path)
    frame = int(accumulation["accumulation_frame"]["repository_count"])
    needed = int(productivity["point_estimate_only"]["development_metadata_candidates_needed_for_542"])
    observed_yield = float(productivity["observed_pilot"]["metadata_to_e2_yield"])
    expected_e2 = math.floor(frame * observed_yield)
    report = {
        "schema_version": "temporal-code-forward-discovery-capacity-report-v1",
        "status": "forward_repository_frame_meets_point_estimate_candidate_capacity",
        "source_sha256": {
            str(combined_path): sha256_file(combined_path),
            str(accumulation_path): sha256_file(accumulation_path),
            str(productivity_path): sha256_file(productivity_path),
        },
        "summary": {
            "combined_discovered_repository_count": combined["summary"]["candidate_count"],
            "frozen_fresh_repository_frame_count": frame,
            "point_estimate_metadata_candidates_needed": needed,
            "repository_frame_to_needed_ratio": frame / needed,
            "expected_e2_at_pilot_yield_if_each_repository_yields_one_candidate": expected_e2,
            "development_e2_target": 542,
            "point_estimate_candidate_capacity_met": frame >= needed,
            "actual_task_candidate_count": 0,
            "actual_e2_count": 0,
        },
        "decision": {
            "structural_repository_frame_blocker_resolved": True,
            "actual_task_distribution_blocker_resolved": False,
            "frame_guarantees_task_target": False,
            "later_snapshot_task_metadata_may_be_collected_under_frozen_rule": True,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
            "next_action": "accumulate later-date task metadata under the frozen 5,000-repository frame",
        },
        "task_metadata_read_from_expanded_frame": False,
        "execution_outcomes_read": False,
        "utility_scope": accumulation["utility_scope"],
        "claim_boundary": "Repository-discovery capacity only; no actual task, E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forward discovery capacity report.")
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--accumulation", type=Path, default=DEFAULT_ACCUMULATION)
    parser.add_argument("--productivity", type=Path, default=DEFAULT_PRODUCTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.combined, args.accumulation, args.productivity, args.output)
    print({"status": report["status"], "summary": report["summary"], "decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
