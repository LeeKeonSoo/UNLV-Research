#!/usr/bin/env python3
"""Collect one immutable retrospective-development task-metadata shard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_SCHEDULE = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retrospective_development_schedule.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_combined.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "retrospective_development_snapshots"


def collect(schedule_path: Path, discovery_path: Path, output_path: Path, shard_index: int, client: Any) -> Dict[str, Any]:
    helper = __import__("110_discover_temporal_code_forward_e2_pilot")
    schedule = load_json(schedule_path)
    discovery = load_json(discovery_path)
    contract = schedule["contract"]
    window = schedule["collection_window"]
    shard = schedule["shards"][shard_index]
    limit = int(contract["retrospective_development_contract"]["maximum_pull_requests_examined_per_repository"])
    candidates = []
    repository_reports = []
    for repository in shard["repository_identities"]:
        errors = []
        try:
            pulls = client.recent_pulls(repository, window["start"], window["end"], limit=limit)
        except RuntimeError as exc:
            pulls = []
            errors.append(str(exc))
        selected = None
        for pull in pulls:
            merge = pull.get("mergeCommit") or {}
            parents = ((merge.get("parents") or {}).get("nodes") or [])
            try:
                paths = client.paths(repository, int(pull["number"]))
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            classification = helper._classify(paths)
            if classification["path_stratum"] not in {"code_and_test", "test_only"} or not parents:
                continue
            source = discovery["candidates"][repository]
            selected = {
                "repository_identity": repository,
                "repository_url": source["repository_url"],
                "license": source["license"],
                "pull_request_number": int(pull["number"]),
                "merge_timestamp": pull["mergedAt"],
                "merge_commit": str(merge.get("oid") or ""),
                "parent_commit": str(parents[0].get("oid") or ""),
                **classification,
                "assigned_split": "retrospective_development",
                "evaluation_authorized_pending_e2_and_quarantine": False,
            }
            candidates.append(selected)
            break
        repository_reports.append(
            {"repository_identity": repository, "pull_count": len(pulls), "selected": selected is not None, "errors": errors}
        )
    candidates.sort(key=lambda row: (row["repository_identity"], row["merge_timestamp"], row["pull_request_number"]))
    report = {
        "schema_version": "temporal-code-retrospective-development-snapshot-shard-v1",
        "status": "immutable_retrospective_snapshot_shard_frozen_before_recipe_or_execution",
        "snapshot_identity": f"retrospective__shard_{shard['shard_id']}",
        "collection_window": window,
        "shard": shard,
        "source_sha256": {str(schedule_path): sha256_file(schedule_path), str(discovery_path): sha256_file(discovery_path)},
        "summary": {
            "repository_count": shard["repository_count"],
            "candidate_count": len(candidates),
            "github_api_requests": client.requests,
            "repository_error_count": sum(bool(row["errors"]) for row in repository_reports),
        },
        "candidates": candidates,
        "repository_reports": repository_reports,
        "recipe_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Retrospective development task metadata only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect one retrospective development shard.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    discovery_module = __import__("64_discover_temporal_code_repositories")
    helper = __import__("110_discover_temporal_code_forward_e2_pilot")
    token, _ = discovery_module.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    output = args.output_dir / f"retrospective__shard_{args.shard_index:03d}.json"
    if output.exists():
        raise SystemExit(f"Immutable retrospective snapshot already exists: {output}")
    report = collect(
        args.schedule,
        args.discovery,
        output,
        args.shard_index,
        helper.Client(token, max(0.0, args.delay_seconds)),
    )
    print({"snapshot_identity": report["snapshot_identity"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
