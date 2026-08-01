#!/usr/bin/env python3
"""Build a planning-only productivity report from the forward E2 pilot."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PILOT = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_pilot_report.json"
DEFAULT_PRIMARY = OUTPUT_DIR / "validation" / "temporal_code_primary_executable_source_assessment.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"


def _needed(target: int, observed_yield: float) -> int | None:
    return math.ceil(target / observed_yield) if observed_yield > 0 else None


def build(pilot_path: Path, primary_path: Path, output_path: Path) -> Dict[str, Any]:
    pilot = load_json(pilot_path)
    primary = load_json(primary_path)
    summary = pilot["summary"]
    metadata_yield = float(summary["metadata_to_e2_yield"])
    execution_yield = float(summary["execution_to_e2_yield"])
    required = int(primary["summary"]["required_primary_task_count"])
    development_target = 542
    confirmatory_target = 541
    failures = Counter(
        row.get("failure_stage") or "task_valid_e2"
        for row in pilot["decisions"]
    )
    report = {
        "schema_version": "temporal-code-forward-e2-productivity-report-v1",
        "status": "forward_e2_acquisition_feasible_but_not_ready_for_utility",
        "source_sha256": {
            str(pilot_path): sha256_file(pilot_path),
            str(primary_path): sha256_file(primary_path),
        },
        "observed_pilot": {
            **summary,
            "failure_stage_counts": dict(sorted(failures.items())),
        },
        "point_estimate_only": {
            "metadata_candidates_needed_for_1083": _needed(required, metadata_yield),
            "execution_attempts_needed_for_1083": _needed(required, execution_yield),
            "development_metadata_candidates_needed_for_542": _needed(development_target, metadata_yield),
            "confirmatory_metadata_candidates_needed_for_541": _needed(confirmatory_target, metadata_yield),
            "development_execution_attempts_needed_for_542": _needed(development_target, execution_yield),
            "confirmatory_execution_attempts_needed_for_541": _needed(confirmatory_target, execution_yield),
        },
        "interpretation": {
            "forward_acquisition_produced_task_valid_e2": summary["task_valid_e2_count"] > 0,
            "pilot_tasks_evaluation_authorized": False,
            "pilot_too_small_for_capacity_commitment": True,
            "estimate_role": "rough infrastructure capacity planning only",
            "inferential_yield_or_capacity_claim_allowed": False,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
            "zero_yield_action": "revise the outcome-independent acquisition pipeline; do not force an infinite collection estimate",
            "next_action": (
                "scale metadata-only discovery and reusable recipe extraction across repositories disjoint from "
                "training, while preserving future development/confirmatory repository and time separation"
            ),
        },
        "utility_scope": pilot["utility_scope"],
        "claim_boundary": (
            "Forward E2 acquisition productivity only; no Utility, curation-benefit, selector, or release claim."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forward E2 pilot productivity report.")
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.pilot, args.primary, args.output)
    print({"status": report["status"], "point_estimate_only": report["point_estimate_only"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
