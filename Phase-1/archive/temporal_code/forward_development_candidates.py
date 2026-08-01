#!/usr/bin/env python3
"""Discover outcome-free task candidates for a forward development snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_snapshot_plan.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_candidates.json"


def discover(plan_path: Path, discovery_path: Path, output_path: Path, client: Any) -> Dict[str, Any]:
    pilot_module = __import__("110_discover_temporal_code_forward_e2_pilot")
    plan = load_json(plan_path)
    discovery = load_json(discovery_path)
    snapshot = plan["snapshot"]
    contract = plan["contract"]
    limit = int(contract["development_acquisition_snapshots"]["maximum_pull_requests_examined_per_repository"])
    candidates = []
    repository_reports = []
    for repository in snapshot["repository_identities"]:
        errors = []
        pulls = []
        try:
            pulls = client.recent_pulls(
                repository,
                snapshot["window_start"],
                snapshot["available_through"],
                limit=limit,
            )
        except RuntimeError as exc:
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
            classification = pilot_module._classify(paths)
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
                "assigned_split": "development",
                "evaluation_authorized_pending_e2_and_quarantine": False,
            }
            candidates.append(selected)
            break
        repository_reports.append(
            {"repository_identity": repository, "pull_count": len(pulls), "selected": selected is not None, "errors": errors}
        )
        print({"repository": repository, "pulls": len(pulls), "selected": selected is not None})
    candidates.sort(key=lambda row: (row["repository_identity"], row["pull_request_number"]))
    report = {
        "schema_version": "temporal-code-forward-development-candidates-v1",
        "status": "frozen_before_project_metadata_or_execution_outcomes",
        "snapshot": snapshot,
        "source_sha256": {str(plan_path): sha256_file(plan_path), str(discovery_path): sha256_file(discovery_path)},
        "summary": {
            "repository_frame_count": len(snapshot["repository_identities"]),
            "candidate_count": len(candidates),
            "code_and_test_count": sum(row["path_stratum"] == "code_and_test" for row in candidates),
            "test_only_count": sum(row["path_stratum"] == "test_only" for row in candidates),
            "github_api_requests": client.requests,
        },
        "candidates": candidates,
        "repository_reports": repository_reports,
        "forbidden_fields_collected": [],
        "training_repository_overlap_count": plan["training_repository_overlap_count"],
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Forward development task candidates only; no E2, Utility, curation-benefit, or release claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover forward development task candidates.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    discovery_module = __import__("64_discover_temporal_code_repositories")
    pilot_module = __import__("110_discover_temporal_code_forward_e2_pilot")
    token, _ = discovery_module.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    report = discover(
        args.plan,
        args.discovery,
        args.output,
        pilot_module.Client(token, max(0.0, args.delay_seconds)),
    )
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
