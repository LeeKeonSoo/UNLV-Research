#!/usr/bin/env python3
"""Freeze a minimal, bounded temporal-code content-fetch smoke plan."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest_authenticated.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_smoke30.json"
DEFAULT_REPRODUCIBILITY = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_smoke30.json"
DEFAULT_ARTIFACTS = OUTPUT_DIR / "temporal_code_collection" / "benchmark_task_artifact_manifest_swebench.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_smoke_fetch_plan.json"


def freeze(
    protocol_path: Path,
    discovery_path: Path,
    enrichment_path: Path,
    reproducibility_path: Path,
    artifacts_path: Path,
    output_path: Path,
    *,
    minimum_merged_prs: int,
) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    discovery = load_json(discovery_path)
    enrichment = load_json(enrichment_path)
    reproducibility = load_json(reproducibility_path)
    artifacts = load_json(artifacts_path)
    allowed_licenses = set(protocol["collection_contract"]["allowed_licenses"])
    benchmark_commits = {
        commit
        for benchmark in artifacts["benchmarks"]
        for rule in benchmark["task_artifact_rules"]
        for commit in rule["commit_oids"]
    }
    reproducible = {
        identity: row
        for identity, row in reproducibility["repositories"].items()
        if row["eligible_for_quarantine_review"]
    }
    candidates = {"train": [], "development": [], "confirmatory": []}
    for identity, row in enrichment["repositories"].items():
        if identity not in reproducible:
            continue
        discovery_row = discovery["candidates"][identity]
        sampled_commits = {
            check["merge_commit"]
            for check in reproducible[identity]["sampled_commit_checks"]
        }
        sampled_commits.update(
            parent
            for check in reproducible[identity]["sampled_commit_checks"]
            for parent in check["parent_commit_identities"]
        )
        blockers = []
        if discovery_row["license"] not in allowed_licenses:
            blockers.append("license_not_allowlisted")
        if int(row["merged_pr_evidence"]["issue_count"]) < minimum_merged_prs:
            blockers.append("insufficient_merged_prs_in_assigned_window")
        if sampled_commits.intersection(benchmark_commits):
            blockers.append("swebench_commit_collision")
        if blockers:
            continue
        candidates[row["assigned_split"]].append(
            {
                "repository_identity": identity,
                "repository_url": discovery_row["repository_url"],
                "license": discovery_row["license"],
                "assigned_split": row["assigned_split"],
                "tree_path_count": row["tree_evidence"]["tree_path_count"],
                "merged_pr_count_in_window": row["merged_pr_evidence"]["issue_count"],
                "sampled_prs": row["merged_pr_evidence"]["samples"],
                "swebench_sampled_commit_collision": False,
            }
        )
    selected = {}
    for split, rows in candidates.items():
        rows.sort(key=lambda item: (item["tree_path_count"], item["repository_identity"]))
        if not rows:
            raise RuntimeError(f"No smoke candidate satisfies the frozen gate for split={split}")
        selected[split] = rows[0]
    plan = {
        "schema_version": "temporal-code-smoke-fetch-plan-v1",
        "status": "frozen_before_content_fetch",
        "protocol_name": protocol["protocol_name"],
        "selection_rule": (
            "For each split, choose the smallest path-count repository among enrichment candidates with "
            "allowlisted license, at least the minimum merged-PR count, reproducible sampled commit identities, "
            "and no sampled SWE-bench commit collision."
        ),
        "minimum_merged_prs": minimum_merged_prs,
        "selected_repositories": selected,
        "content_fetch_limits": {
            "maximum_pull_requests_per_repository": 2,
            "maximum_changed_files_per_pull_request": 50,
            "maximum_file_bytes": 524288,
            "allowed_file_suffixes": [".py", ".md", ".rst", ".toml", ".cfg", ".ini", ".txt"],
            "issue_and_pull_request_prose": "do_not_fetch_for_training_payload",
            "binary_generated_vendor_lock_files": "exclude",
        },
        "required_during_fetch": [
            "record parent and merge commit identities",
            "scan fetched text for secrets and PII before persistence to release candidates",
            "retain raw fetched content only under generated-output paths excluded from Git",
            "run benchmark commit and normalized-hash quarantine before generic Stage 0",
            "do not treat smoke repositories as frozen full-corpus repositories",
        ],
        "frozen_repository_manifest_status": "not_frozen_smoke_only",
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Bounded smoke content-fetch plan only; no repository is approved for the full experiment.",
    }
    save_json(output_path, plan)
    lines = ["# Temporal Code Smoke Fetch Plan", "", f"Status: `{plan['status']}`", ""]
    for split, row in selected.items():
        lines.append(
            f"- {split}: `{row['repository_identity']}` "
            f"(paths={row['tree_path_count']}, merged_prs={row['merged_pr_count_in_window']})"
        )
    lines.extend(["", "## Claim Boundary", "", plan["claim_boundary"], ""])
    output_path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze temporal-code smoke fetch plan.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-merged-prs", type=int, default=5)
    args = parser.parse_args()
    plan = freeze(
        args.protocol,
        args.discovery,
        args.enrichment,
        args.reproducibility,
        args.artifacts,
        args.output,
        minimum_merged_prs=max(1, args.minimum_merged_prs),
    )
    print({split: row["repository_identity"] for split, row in plan["selected_repositories"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
