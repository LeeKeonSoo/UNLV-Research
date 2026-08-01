#!/usr/bin/env python3
"""Build the current retrospective development operating status."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_LEDGER = COLLECTION / "temporal_code_retrospective_combined_candidate_ledger.json"
DEFAULT_EXPANSION = COLLECTION / "temporal_code_retrospective_expansion_schedule.json"
DEFAULT_E2_DIR = OUTPUT_DIR / "validation"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_retrospective_operations_status.json"


def build(ledger_path: Path, expansion_path: Path, e2_paths: List[Path], output_path: Path) -> Dict[str, Any]:
    ledger = load_json(ledger_path)
    expansion = load_json(expansion_path)
    e2_reports = [load_json(path) for path in sorted(e2_paths)]
    target = int(ledger["summary"]["development_target_valid_e2_count"])
    valid_e2 = sum(int(report["summary"]["task_valid_e2_count"]) for report in e2_reports)
    attempts = sum(int(report["summary"]["execution_candidate_count"]) for report in e2_reports)
    candidates = int(ledger["summary"]["candidate_count"])
    metadata_complete = bool(ledger["summary"]["metadata_collection_complete"])
    quarantined_authorized = 0
    report = {
        "schema_version": "temporal-code-retrospective-operations-status-v1",
        "status": (
            "retrospective_collection_waiting_for_expansion_metadata"
            if not metadata_complete
            else "retrospective_collection_ready_for_remaining_e2_batches"
        ),
        "source_sha256": {
            str(ledger_path): sha256_file(ledger_path),
            str(expansion_path): sha256_file(expansion_path),
            **{str(path): sha256_file(path) for path in sorted(e2_paths)},
        },
        "summary": {
            "scheduled_repository_count": ledger["summary"]["scheduled_repository_count"],
            "snapshot_count": ledger["summary"]["snapshot_count"],
            "expected_snapshot_count": ledger["summary"]["expected_snapshot_count"],
            "candidate_count": candidates,
            "e2_execution_attempt_count": attempts,
            "task_valid_e2_count": valid_e2,
            "development_target_valid_e2_count": target,
            "valid_e2_gap": max(0, target - valid_e2),
            "contamination_quarantine_authorized_count": quarantined_authorized,
        },
        "gates": {
            "expansion_schedule_frozen": expansion["status"]
            == "frozen_after_first_e2_batch_and_before_remaining_repository_task_metadata",
            "metadata_collection_complete": metadata_complete,
            "combined_candidate_ledger_frozen": True,
            "recipe_freeze_may_continue": candidates > attempts,
            "e2_execution_may_continue": candidates > attempts,
            "contamination_quarantine_may_start": valid_e2 >= target,
            "development_utility_may_start": valid_e2 >= target and quarantined_authorized >= target,
            "confirmatory_outcomes_read": False,
        },
        "next_action": (
            "collect the frozen retrospective expansion metadata shards"
            if not metadata_complete
            else "freeze and execute the next outcome-independent strict E2 recipe batch"
        ),
        "operational_commands": {
            "collect_expansion": (
                "conda run --no-capture-output -n research python 129_run_temporal_code_retrospective_collection.py "
                "--schedule outputs\\temporal_code_collection\\temporal_code_retrospective_expansion_schedule.json "
                "--output-dir outputs\\temporal_code_collection\\retrospective_expansion_snapshots --delay-seconds 0.1"
            ),
            "rebuild_combined_ledger": (
                "conda run --no-capture-output -n research python "
                "132_build_temporal_code_retrospective_combined_ledger.py"
            ),
            "refresh_status": (
                "conda run --no-capture-output -n research python "
                "133_build_temporal_code_retrospective_operations_status.py"
            ),
        },
        "confirmatory_outcomes_read": False,
        "utility_scope": ledger["utility_scope"],
        "claim_boundary": "Retrospective operating readiness only; no Utility, selector, curation-benefit, or release claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrospective development operating status.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--expansion", type=Path, default=DEFAULT_EXPANSION)
    parser.add_argument("--e2-dir", type=Path, default=DEFAULT_E2_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = sorted(args.e2_dir.glob("temporal_code_retrospective_e2_batch_*_report.json"))
    report = build(args.ledger, args.expansion, paths, args.output)
    print({"status": report["status"], "summary": report["summary"], "next_action": report["next_action"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
