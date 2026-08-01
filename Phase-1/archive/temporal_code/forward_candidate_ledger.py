#!/usr/bin/env python3
"""Build a cumulative one-task-per-repository forward candidate ledger."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_SCHEDULE = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json"
DEFAULT_SNAPSHOT_DIR = OUTPUT_DIR / "temporal_code_collection" / "forward_development_snapshots"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_candidate_ledger.json"


def build(schedule_path: Path, snapshot_paths: List[Path], output_path: Path) -> Dict[str, Any]:
    schedule = load_json(schedule_path)
    rows: Dict[str, Dict[str, Any]] = {}
    duplicate_candidate_count = 0
    snapshot_summaries = []
    for path in sorted(snapshot_paths):
        snapshot = load_json(path)
        snapshot_summaries.append(
            {"path": str(path), "snapshot_identity": snapshot["snapshot_identity"], "summary": snapshot["summary"]}
        )
        for row in snapshot["candidates"]:
            repository = row["repository_identity"]
            existing = rows.get(repository)
            choice = (row["merge_timestamp"], int(row["pull_request_number"]), row["merge_commit"])
            if existing is not None:
                duplicate_candidate_count += 1
                existing_choice = (
                    existing["merge_timestamp"],
                    int(existing["pull_request_number"]),
                    existing["merge_commit"],
                )
                if choice >= existing_choice:
                    continue
            rows[repository] = {**row, "candidate_ledger_frozen": True}
    candidates = sorted(rows.values(), key=lambda row: (row["merge_timestamp"], row["repository_identity"]))
    target = int(schedule["contract"]["future_primary_acquisition"]["development_window"]["target_task_count"])
    report = {
        "schema_version": "temporal-code-forward-candidate-ledger-v1",
        "status": "candidate_ledger_frozen_before_recipe_or_execution",
        "source_sha256": {
            str(schedule_path): sha256_file(schedule_path),
            **{str(path): sha256_file(path) for path in sorted(snapshot_paths)},
        },
        "summary": {
            "snapshot_count": len(snapshot_paths),
            "candidate_count": len(candidates),
            "unique_repository_count": len(rows),
            "duplicate_candidate_count": duplicate_candidate_count,
            "development_target_task_count": target,
            "candidate_target_met": len(candidates) >= target,
        },
        "candidates": candidates,
        "snapshot_summaries": snapshot_summaries,
        "recipe_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": schedule["utility_scope"],
        "claim_boundary": "Candidate ledger only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cumulative forward candidate ledger.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.schedule, sorted(args.snapshot_dir.glob("*.json")), args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
