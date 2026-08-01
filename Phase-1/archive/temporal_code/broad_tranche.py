#!/usr/bin/env python3
"""Freeze a deterministic broad-manifest content-fetch tranche."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_broad_tranche_v1.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_tranche_plan.json"


def _quantile_indices(size: int, count: int) -> List[int]:
    if count <= 0 or count > size:
        raise ValueError(f"Invalid tranche count={count} for size={size}")
    if count == 1:
        return [size // 2]
    return [math.floor(index * (size - 1) / (count - 1)) for index in range(count)]


def freeze(contract_path: Path, manifest_path: Path, output_path: Path) -> Dict[str, Any]:
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    if manifest["status"] != "frozen_before_broad_content_fetch":
        raise RuntimeError("Broad repository manifest must be frozen before tranche selection.")
    repositories = manifest["repositories"]
    selected: Dict[str, List[Dict[str, Any]]] = {}
    selected_identities = set()
    for split, count in contract["repositories_per_split"].items():
        rows = sorted(
            (row for row in repositories.values() if row["assigned_split"] == split),
            key=lambda row: (int(row["tree_path_count"]), row["repository_identity"]),
        )
        indices = _quantile_indices(len(rows), int(count))
        selected[split] = []
        for rank_index in indices:
            row = dict(rows[rank_index])
            row["split_rank_index"] = rank_index
            row["split_candidate_count"] = len(rows)
            row["selection_rank_fraction"] = rank_index / max(1, len(rows) - 1)
            row["sampled_prs"] = row["sampled_prs"][: int(contract["maximum_pull_requests_per_repository"])]
            selected[split].append(row)
            selected_identities.add(row["repository_identity"])
    report = {
        "schema_version": "temporal-code-broad-tranche-plan-v1",
        "status": "frozen_before_tranche_content_fetch",
        "contract": contract,
        "source_manifest_sha256": sha256_file(manifest_path),
        "summary": {
            "repository_count": len(selected_identities),
            "split_counts": {split: len(rows) for split, rows in selected.items()},
            "maximum_bundle_count": sum(
                len(row["sampled_prs"]) for rows in selected.values() for row in rows
            ),
        },
        "content_fetch_limits": manifest["freeze_contract"]["content_fetch_limits"],
        "selected_repositories": selected,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a deterministic broad temporal-code tranche.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.manifest, args.output)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
