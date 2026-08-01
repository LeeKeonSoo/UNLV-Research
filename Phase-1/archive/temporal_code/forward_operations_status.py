#!/usr/bin/env python3
"""Build forward-development operations readiness and next-action status."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_SCHEDULE = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json"
DEFAULT_LEDGER = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_candidate_ledger.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_forward_operations_status.json"


def build(schedule_path: Path, ledger_path: Path, output_path: Path) -> Dict[str, Any]:
    schedule = load_json(schedule_path)
    ledger = load_json(ledger_path)
    target = int(schedule["contract"]["future_primary_acquisition"]["development_window"]["target_task_count"])
    candidates = int(ledger["summary"]["candidate_count"])
    report = {
        "schema_version": "temporal-code-forward-operations-status-v1",
        "status": "forward_collection_operational_waiting_for_later_date_tasks",
        "source_sha256": {str(schedule_path): sha256_file(schedule_path), str(ledger_path): sha256_file(ledger_path)},
        "summary": {
            "repository_count": schedule["summary"]["repository_count"],
            "shard_count": schedule["summary"]["shard_count"],
            "snapshot_count": ledger["summary"]["snapshot_count"],
            "candidate_count": candidates,
            "candidate_target": target,
            "candidate_gap": max(0, target - candidates),
            "candidate_target_met": candidates >= target,
            "recipe_ready_candidate_count": candidates,
            "verified_e2_count": 0,
        },
        "gates": {
            "repository_schedule_frozen": True,
            "snapshot_artifacts_immutable": True,
            "candidate_ledger_frozen_before_recipe": True,
            "recipe_freeze_may_start": candidates > 0,
            "e2_execution_may_start": False,
            "development_utility_may_start": False,
            "confirmatory_outcomes_read": False,
        },
        "next_action": (
            "collect later-date immutable snapshot shards and rebuild the candidate ledger"
            if candidates == 0
            else "freeze outcome-independent project-metadata recipes for the next unprocessed candidate batch"
        ),
        "operational_commands": {
            "refresh": "conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py --action refresh",
            "collect_shard_template": (
                "conda run --no-capture-output -n research python 125_run_temporal_code_forward_operations.py "
                "--action collect --available-through YYYY-MM-DD --shard-index N"
            ),
            "freeze_recipe_batch": (
                "conda run --no-capture-output -n research python 126_freeze_temporal_code_forward_recipe_batch.py "
                "--start-index N"
            ),
            "verify_recipe_batch": (
                "conda run --no-capture-output -n research python 112_verify_temporal_code_forward_e2_pilot.py "
                "--recipes outputs\\temporal_code_collection\\temporal_code_forward_recipe_batch.json "
                "--work-dir outputs\\temporal_code_collection\\forward_development_e2_work "
                "--output outputs\\validation\\temporal_code_forward_development_e2_batch_report.json"
            ),
        },
        "utility_scope": schedule["utility_scope"],
        "claim_boundary": "Forward collection operations readiness only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build forward operations status.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.schedule, args.ledger, args.output)
    print({"status": report["status"], "summary": report["summary"], "next_action": report["next_action"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
