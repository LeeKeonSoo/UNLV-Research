#!/usr/bin/env python3
"""Freeze the broad temporal-code repository manifest before content fetch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_change import normalize_repository_identity


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_FREEZE = Path("configs") / "temporal_code_broad_collection_freeze_v1.json"
DEFAULT_BENCHMARK_SEED = Path("validation") / "fixtures" / "temporal_code_benchmark_quarantine_seed.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest_authenticated.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_broad499.json"
DEFAULT_REPRODUCIBILITY = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_broad499.json"
DEFAULT_READINESS = OUTPUT_DIR / "temporal_code_collection" / "collection_readiness_report_broad499.json"
DEFAULT_ARTIFACTS = OUTPUT_DIR / "temporal_code_collection" / "benchmark_task_artifact_manifest_swebench.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"


def freeze(
    protocol_path: Path,
    freeze_path: Path,
    benchmark_seed_path: Path,
    discovery_path: Path,
    enrichment_path: Path,
    reproducibility_path: Path,
    readiness_path: Path,
    artifacts_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    freeze_contract = load_json(freeze_path)
    benchmark_seed = load_json(benchmark_seed_path)
    discovery = load_json(discovery_path)
    enrichment = load_json(enrichment_path)
    reproducibility = load_json(reproducibility_path)
    readiness = load_json(readiness_path)
    artifacts = load_json(artifacts_path)
    if readiness["status"] not in {
        "ready_to_freeze_repository_manifest",
        "broad_repository_manifest_frozen",
    } or readiness["blockers"]:
        raise RuntimeError("Broad repository manifest cannot freeze before readiness blockers are cleared.")

    allowed_licenses = set(protocol["collection_contract"]["allowed_licenses"])
    benchmark_repositories = {
        normalize_repository_identity(identity)
        for entry in benchmark_seed["entries"]
        for identity in entry.get("repository_patterns") or []
    }
    benchmark_commits = {
        commit
        for benchmark in artifacts["benchmarks"]
        for rule in benchmark["task_artifact_rules"]
        for commit in rule["commit_oids"]
    }
    minimum_merged_prs = int(freeze_contract["minimum_merged_prs_in_assigned_window"])
    included: Dict[str, Any] = {}
    exclusions: Dict[str, Any] = {}
    reproducible = reproducibility["repositories"]
    for identity, row in enrichment["repositories"].items():
        blockers = []
        discovery_row = discovery["candidates"][identity]
        reproducibility_row = reproducible.get(identity)
        if row["eligible_for_reproducibility_probe"] is not True:
            blockers.extend(row["blockers"])
        if not reproducibility_row or reproducibility_row["eligible_for_quarantine_review"] is not True:
            blockers.append("commit_reproducibility_gate_failed")
        if discovery_row["license"] not in allowed_licenses:
            blockers.append("license_not_allowlisted")
        if identity in benchmark_repositories:
            blockers.append("benchmark_source_repository")
        if int(row["merged_pr_evidence"]["issue_count"]) < minimum_merged_prs:
            blockers.append("insufficient_merged_prs_in_assigned_window")
        sampled_commits = set()
        if reproducibility_row:
            sampled_commits.update(
                check["merge_commit"] for check in reproducibility_row["sampled_commit_checks"]
            )
            sampled_commits.update(
                parent
                for check in reproducibility_row["sampled_commit_checks"]
                for parent in check["parent_commit_identities"]
            )
        if sampled_commits.intersection(benchmark_commits):
            blockers.append("benchmark_commit_collision")
        if blockers:
            exclusions[identity] = {"blockers": sorted(set(blockers))}
            continue
        included[identity] = {
            "repository_identity": identity,
            "repository_url": discovery_row["repository_url"],
            "license": discovery_row["license"],
            "assigned_split": row["assigned_split"],
            "tree_path_count": row["tree_evidence"]["tree_path_count"],
            "merged_pr_count_in_window": row["merged_pr_evidence"]["issue_count"],
            "sampled_prs": row["merged_pr_evidence"]["samples"],
            "sampled_commit_collision": False,
            "membership_is_training_approval": False,
        }
    included = dict(
        sorted(included.items(), key=lambda item: (item[1]["assigned_split"], item[0]))
    )
    split_counts = {
        split: sum(row["assigned_split"] == split for row in included.values())
        for split in ("train", "development", "confirmatory")
    }
    if not all(split_counts.values()):
        raise RuntimeError(f"Broad frozen manifest requires every split: {split_counts}")
    source_paths = [
        protocol_path,
        freeze_path,
        benchmark_seed_path,
        discovery_path,
        enrichment_path,
        reproducibility_path,
        artifacts_path,
    ]
    report = {
        "schema_version": "temporal-code-broad-repository-manifest-v1",
        "status": "frozen_before_broad_content_fetch",
        "protocol_name": protocol["protocol_name"],
        "freeze_contract": freeze_contract,
        "source_artifact_sha256": {str(path): sha256_file(path) for path in source_paths},
        "summary": {
            "discovery_candidates": discovery["summary"]["candidate_count"],
            "enriched_repositories": enrichment["summary"]["repository_count"],
            "reproducibility_pass_repositories": reproducibility["summary"][
                "eligible_for_quarantine_review_count"
            ],
            "frozen_repository_count": len(included),
            "excluded_repository_count": len(exclusions),
            "split_counts": split_counts,
        },
        "repositories": included,
        "exclusions": dict(sorted(exclusions.items())),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": freeze_contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the broad temporal-code repository manifest.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze-contract", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--benchmark-seed", type=Path, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(
        args.protocol,
        args.freeze_contract,
        args.benchmark_seed,
        args.discovery,
        args.enrichment,
        args.reproducibility,
        args.readiness,
        args.artifacts,
        args.output,
    )
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
