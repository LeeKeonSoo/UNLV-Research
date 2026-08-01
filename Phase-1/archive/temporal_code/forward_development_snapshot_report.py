#!/usr/bin/env python3
"""Build the first actual forward-development acquisition snapshot report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_snapshot_plan.json"
DEFAULT_CANDIDATES = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_candidates.json"
DEFAULT_PRODUCTIVITY = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_development_snapshot_report.json"


def build(plan_path: Path, candidates_path: Path, productivity_path: Path, output_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    candidates = load_json(candidates_path)
    productivity = load_json(productivity_path)
    count = int(candidates["summary"]["candidate_count"])
    report = {
        "schema_version": "temporal-code-forward-development-snapshot-report-v1",
        "status": "forward_development_snapshot_complete_no_candidates" if count == 0 else "forward_development_candidates_frozen",
        "source_sha256": {
            str(plan_path): sha256_file(plan_path),
            str(candidates_path): sha256_file(candidates_path),
            str(productivity_path): sha256_file(productivity_path),
        },
        "summary": {
            "window_start": plan["snapshot"]["window_start"],
            "available_through": plan["snapshot"]["available_through"],
            "fresh_repository_frame_count": plan["snapshot"]["repository_frame_count"],
            "metadata_candidate_count": count,
            "training_repository_overlap_count": candidates["training_repository_overlap_count"],
            "execution_recipe_count": 0,
            "task_valid_e2_count": 0,
        },
        "decision": {
            "zero_candidates_is_valid_snapshot_evidence": count == 0,
            "retroactively_expand_same_snapshot_after_candidate_outcome": False,
            "candidate_recipe_or_execution_may_start": count > 0,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
            "next_action": (
                "freeze the next-date higher-capacity accumulation snapshot before reading its task metadata; "
                "deduplicate prior repository and pull-request identities; continue abstaining"
            ),
            "capacity_basis_for_next_snapshot": productivity["point_estimate_only"],
        },
        "execution_outcomes_read": False,
        "utility_scope": candidates["utility_scope"],
        "claim_boundary": "Forward acquisition snapshot accounting only; no E2, Utility, curation, or release claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forward development snapshot report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--productivity", type=Path, default=DEFAULT_PRODUCTIVITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.plan, args.candidates, args.productivity, args.output)
    print({"status": report["status"], "summary": report["summary"], "decision": report["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
