#!/usr/bin/env python3
"""Audit whether continued strict E2 execution can plausibly meet the target."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_LEDGER = COLLECTION / "temporal_code_retrospective_combined_candidate_ledger.json"
DEFAULT_E2_DIR = OUTPUT_DIR / "validation"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_retrospective_e2_capacity_audit.json"
ONE_SIDED_95_Z = 1.6448536269514722


def _wilson_upper(successes: int, trials: int, z: float = ONE_SIDED_95_Z) -> float:
    if trials == 0:
        return 1.0
    p = successes / trials
    denominator = 1 + z * z / trials
    center = p + z * z / (2 * trials)
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return min(1.0, (center + radius) / denominator)


def build(ledger_path: Path, e2_paths: list[Path], output_path: Path) -> Dict[str, Any]:
    ledger = load_json(ledger_path)
    reports = [load_json(path) for path in sorted(e2_paths)]
    attempts = sum(int(report["summary"]["execution_candidate_count"]) for report in reports)
    valid = sum(int(report["summary"]["task_valid_e2_count"]) for report in reports)
    candidate_count = int(ledger["summary"]["candidate_count"])
    target = int(ledger["summary"]["development_target_valid_e2_count"])
    remaining = max(0, candidate_count - attempts)
    upper = _wilson_upper(valid, attempts)
    projected_upper_total = valid + math.floor(remaining * upper)
    failures = Counter(
        row.get("failure_stage") or "task_valid_e2"
        for report in reports
        for row in report["decisions"]
    )
    target_reached = valid >= target
    capacity_still_plausible = projected_upper_total >= target
    report = {
        "schema_version": "temporal-code-retrospective-e2-capacity-audit-v1",
        "status": (
            "retrospective_valid_e2_target_reached"
            if target_reached
            else (
                "retrospective_strict_e2_execution_should_continue"
                if capacity_still_plausible
                else "retrospective_strict_e2_capacity_infeasible_at_frozen_confidence_rule"
            )
        ),
        "source_sha256": {
            str(ledger_path): sha256_file(ledger_path),
            **{str(path): sha256_file(path) for path in sorted(e2_paths)},
        },
        "observed": {
            "candidate_count": candidate_count,
            "execution_attempt_count": attempts,
            "task_valid_e2_count": valid,
            "observed_task_valid_e2_rate": valid / attempts if attempts else 0.0,
            "remaining_candidate_count": remaining,
            "failure_stage_counts": dict(sorted(failures.items())),
        },
        "frozen_stopping_rule": {
            "development_target_valid_e2_count": target,
            "rate_bound": "one-sided 95% Wilson upper bound over repository-level task-valid E2 outcomes",
            "one_sided_z": ONE_SIDED_95_Z,
            "task_valid_e2_rate_upper_bound": upper,
            "projected_upper_total_valid_e2": projected_upper_total,
            "stop_as_capacity_infeasible_when": "current valid E2 plus remaining candidates times rate upper bound is below target",
            "recipe_or_task_validity_rule_changes": "forbidden",
        },
        "decision": {
            "target_reached": target_reached,
            "capacity_still_plausible": capacity_still_plausible,
            "strict_e2_execution_may_continue": not target_reached and capacity_still_plausible,
            "contamination_quarantine_may_start": target_reached,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
        },
        "utility_scope": ledger["utility_scope"],
        "claim_boundary": "Strict E2 execution capacity only; no Utility, selector, curation-benefit, or release claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrospective E2 capacity audit.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = sorted(args.e2_dir.glob("temporal_code_retrospective_e2_batch_*_report.json"))
    report = build(args.ledger, paths, args.output)
    print({"status": report["status"], "observed": report["observed"], "stopping": report["frozen_stopping_rule"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
