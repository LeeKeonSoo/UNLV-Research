#!/usr/bin/env python3
"""Build one retrospective ledger across initial and expansion snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_INITIAL_SCHEDULE = COLLECTION / "temporal_code_retrospective_development_schedule.json"
DEFAULT_EXPANSION_SCHEDULE = COLLECTION / "temporal_code_retrospective_expansion_schedule.json"
DEFAULT_INITIAL_DIR = COLLECTION / "retrospective_development_snapshots"
DEFAULT_EXPANSION_DIR = COLLECTION / "retrospective_expansion_snapshots"
DEFAULT_OUTPUT = COLLECTION / "temporal_code_retrospective_combined_candidate_ledger.json"


def _scheduled_repositories(schedule: Dict[str, Any]) -> set[str]:
    return {identity for shard in schedule["shards"] for identity in shard["repository_identities"]}


def build(
    initial_schedule_path: Path,
    expansion_schedule_path: Path,
    snapshot_paths: Iterable[Path],
    output_path: Path,
) -> Dict[str, Any]:
    initial = load_json(initial_schedule_path)
    expansion = load_json(expansion_schedule_path)
    initial_ids = _scheduled_repositories(initial)
    expansion_ids = _scheduled_repositories(expansion)
    overlap = initial_ids.intersection(expansion_ids)
    if overlap:
        raise ValueError("Initial and expansion retrospective schedules must be repository-disjoint.")
    allowed = initial_ids.union(expansion_ids)
    rows: Dict[str, Dict[str, Any]] = {}
    duplicate_candidate_count = 0
    snapshot_summaries: List[Dict[str, Any]] = []
    paths = sorted(snapshot_paths)
    for path in paths:
        snapshot = load_json(path)
        snapshot_summaries.append(
            {"path": str(path), "snapshot_identity": snapshot["snapshot_identity"], "summary": snapshot["summary"]}
        )
        for row in snapshot["candidates"]:
            repository = row["repository_identity"]
            if repository not in allowed:
                raise ValueError(f"Snapshot candidate is outside the frozen retrospective schedules: {repository}")
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
    target = int(initial["contract"]["retrospective_development_contract"]["development_target_valid_e2_count"])
    expected_snapshots = int(initial["summary"]["shard_count"]) + int(expansion["summary"]["shard_count"])
    report = {
        "schema_version": "temporal-code-retrospective-combined-candidate-ledger-v1",
        "status": "combined_retrospective_candidate_ledger_frozen_before_recipe_or_execution",
        "source_sha256": {
            str(initial_schedule_path): sha256_file(initial_schedule_path),
            str(expansion_schedule_path): sha256_file(expansion_schedule_path),
            **{str(path): sha256_file(path) for path in paths},
        },
        "summary": {
            "scheduled_repository_count": len(allowed),
            "initial_snapshot_count": sum("retrospective_development_snapshots" in str(path) for path in paths),
            "expansion_snapshot_count": sum("retrospective_expansion_snapshots" in str(path) for path in paths),
            "snapshot_count": len(paths),
            "expected_snapshot_count": expected_snapshots,
            "metadata_collection_complete": len(paths) == expected_snapshots,
            "candidate_count": len(candidates),
            "unique_repository_count": len(rows),
            "duplicate_candidate_count": duplicate_candidate_count,
            "development_target_valid_e2_count": target,
        },
        "candidates": candidates,
        "snapshot_summaries": snapshot_summaries,
        "recipe_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": initial["utility_scope"],
        "claim_boundary": "Combined retrospective candidate ledger only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined retrospective candidate ledger.")
    parser.add_argument("--initial-schedule", type=Path, default=DEFAULT_INITIAL_SCHEDULE)
    parser.add_argument("--expansion-schedule", type=Path, default=DEFAULT_EXPANSION_SCHEDULE)
    parser.add_argument("--initial-dir", type=Path, default=DEFAULT_INITIAL_DIR)
    parser.add_argument("--expansion-dir", type=Path, default=DEFAULT_EXPANSION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = list(args.initial_dir.glob("*.json")) + list(args.expansion_dir.glob("*.json"))
    report = build(args.initial_schedule, args.expansion_schedule, paths, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
