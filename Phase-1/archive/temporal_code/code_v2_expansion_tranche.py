#!/usr/bin/env python3
"""Freeze a code-domain v2 expansion tranche before content fetch."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONFIG = Path("configs") / "code_domain_v2_expansion_tranche_v1.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_PATH_METADATA = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_pr_path_metadata.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "code_domain_v2_expansion_tranche_plan.json"


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


def _plan_repositories(plan: Dict[str, Any]) -> Iterable[str]:
    for value in plan.get("selected_repositories", {}).values():
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            identity = row.get("repository_identity")
            if identity:
                yield str(identity)


def _excluded_repositories(config: Dict[str, Any]) -> set[str]:
    excluded = set(str(value) for value in config.get("excluded_repository_identities") or [])
    for raw_path in config.get("exclude_repository_plans") or []:
        path = Path(str(raw_path))
        if path.exists():
            excluded.update(_plan_repositories(load_json(path)))
    return excluded


def freeze(config_path: Path, manifest_path: Path, path_metadata_path: Path, output_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    manifest = load_json(manifest_path)
    path_metadata = load_json(path_metadata_path)
    priority = list(config["representative_pull_request_priority"])
    excluded = _excluded_repositories(config)
    candidates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        split: {stratum: [] for stratum in priority}
        for split in config["required_repositories_by_split_and_stratum"]
    }
    for identity, metadata_row in path_metadata["repositories"].items():
        if identity in excluded:
            continue
        representative = _representative_pull(metadata_row, priority)
        if representative is None:
            continue
        source = manifest["repositories"].get(identity)
        if not source:
            continue
        row = dict(source)
        row["sampled_prs"] = [
            sample for sample in source["sampled_prs"] if int(sample["number"]) == int(representative["number"])
        ]
        if not row["sampled_prs"]:
            continue
        row["path_stratum"] = representative["path_stratum"]
        row["path_metadata_complete"] = True
        row["excluded_prior_v2_tranche"] = False
        candidates[row["assigned_split"]][row["path_stratum"]].append(row)

    blockers = []
    selected: Dict[str, List[Dict[str, Any]]] = {split: [] for split in candidates}
    availability = {}
    for split, required in config["required_repositories_by_split_and_stratum"].items():
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
    identities = [row["repository_identity"] for rows in selected.values() for row in rows]
    if len(identities) != len(set(identities)):
        blockers.append("repository_reused_across_selected_bundles")
    selected_counts = {
        split: dict(sorted(Counter(row["path_stratum"] for row in rows).items()))
        for split, rows in selected.items()
    }
    status = "frozen_before_expansion_content_fetch" if not blockers else config["failure_decision"]
    report = {
        "schema_version": "code-domain-v2-expansion-tranche-plan-v1",
        "status": status,
        "contract": config,
        "source_sha256": {
            str(config_path): sha256_file(config_path),
            str(manifest_path): sha256_file(manifest_path),
            str(path_metadata_path): sha256_file(path_metadata_path),
            **{
                str(Path(path)): sha256_file(Path(path))
                for path in config.get("exclude_repository_plans") or []
                if Path(path).exists()
            },
        },
        "summary": {
            "excluded_repository_count": len(excluded),
            "repository_count": len(set(identities)),
            "maximum_bundle_count": len(identities),
            "selected_counts": selected_counts,
            "candidate_availability_after_exclusion": availability,
            "blockers": sorted(set(blockers)),
        },
        "content_fetch_limits": manifest["freeze_contract"]["content_fetch_limits"],
        "selected_repositories": selected,
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain v2 expansion tranche.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-metadata", type=Path, default=DEFAULT_PATH_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.config, args.manifest, args.path_metadata, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0 if report["status"] == "frozen_before_expansion_content_fetch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
