#!/usr/bin/env python3
"""Freeze a fresh path-stratified tranche before fetching file content."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_path_stratified_tranche_v1.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_PATH_METADATA = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_pr_path_metadata.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_path_stratified_tranche_plan.json"


def _quantile_indices(size: int, count: int) -> List[int]:
    if count <= 0 or count > size:
        raise ValueError(f"Invalid tranche count={count} for size={size}")
    if count == 1:
        return [size // 2]
    return [math.floor(index * (size - 1) / (count - 1)) for index in range(count)]


def _representative_pull(repository: Dict[str, Any], priority: List[str]) -> Dict[str, Any] | None:
    eligible = [
        row
        for row in repository["pull_requests"]
        if row["path_metadata_complete"] is True and row["path_stratum"] in priority
    ]
    if not eligible:
        return None
    rank = {stratum: index for index, stratum in enumerate(priority)}
    return min(eligible, key=lambda row: (rank[row["path_stratum"]], int(row["number"])))


def freeze(
    contract_path: Path,
    manifest_path: Path,
    path_metadata_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    path_metadata = load_json(path_metadata_path)
    priority = list(contract["representative_pull_request_priority"])
    candidates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        split: {stratum: [] for stratum in priority}
        for split in contract["required_repositories_by_split_and_stratum"]
    }
    for identity, metadata_row in path_metadata["repositories"].items():
        representative = _representative_pull(metadata_row, priority)
        if representative is None:
            continue
        source = manifest["repositories"][identity]
        row = dict(source)
        row["sampled_prs"] = [
            sample for sample in source["sampled_prs"] if int(sample["number"]) == int(representative["number"])
        ]
        row["path_stratum"] = representative["path_stratum"]
        row["path_metadata_complete"] = True
        candidates[row["assigned_split"]][row["path_stratum"]].append(row)

    blockers = []
    selected: Dict[str, List[Dict[str, Any]]] = {split: [] for split in candidates}
    availability = {}
    for split, required in contract["required_repositories_by_split_and_stratum"].items():
        availability[split] = {}
        for stratum, count_value in required.items():
            count = int(count_value)
            rows = sorted(
                candidates[split][stratum],
                key=lambda row: (int(row["tree_path_count"]), row["repository_identity"]),
            )
            availability[split][stratum] = len(rows)
            if len(rows) < count:
                blockers.append(f"insufficient_{split}_{stratum}")
                continue
            for index in _quantile_indices(len(rows), count):
                selected[split].append(rows[index])
    identities = [
        row["repository_identity"] for split_rows in selected.values() for row in split_rows
    ]
    if len(identities) != len(set(identities)):
        blockers.append("repository_reused_across_selected_bundles")
    selected_counts = {
        split: dict(sorted(Counter(row["path_stratum"] for row in rows).items()))
        for split, rows in selected.items()
    }
    status = "frozen_before_tranche_content_fetch" if not blockers else contract["failure_decision"]
    report = {
        "schema_version": "temporal-code-path-stratified-tranche-plan-v1",
        "status": status,
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(manifest_path): sha256_file(manifest_path),
            str(path_metadata_path): sha256_file(path_metadata_path),
        },
        "summary": {
            "repository_count": len(set(identities)),
            "maximum_bundle_count": len(identities),
            "selected_counts": selected_counts,
            "candidate_availability": availability,
            "blockers": sorted(set(blockers)),
        },
        "content_fetch_limits": manifest["freeze_contract"]["content_fetch_limits"],
        "selected_repositories": selected,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a path-stratified temporal-code tranche.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-metadata", type=Path, default=DEFAULT_PATH_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.manifest, args.path_metadata, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0 if report["status"] == "frozen_before_tranche_content_fetch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
