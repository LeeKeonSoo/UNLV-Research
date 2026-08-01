#!/usr/bin/env python3
"""Freeze untouched confirmatory executable-evaluation candidates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_confirmatory_executable_expansion_v1.json"
DEFAULT_MANIFEST = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
DEFAULT_PATH_METADATA = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_pr_path_metadata.json"
DEFAULT_EXCLUDED_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_path_stratified_tranche_plan_v2.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_confirmatory_execution_expansion_plan.json"


def _quantile_indices(size: int, count: int) -> List[int]:
    if count <= 0 or count > size:
        raise ValueError(f"Invalid expansion count={count} for size={size}")
    if count == 1:
        return [size // 2]
    return [math.floor(index * (size - 1) / (count - 1)) for index in range(count)]


def freeze(
    contract_path: Path,
    manifest_path: Path,
    path_metadata_path: Path,
    excluded_plan_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    metadata = load_json(path_metadata_path)
    excluded_plan = load_json(excluded_plan_path)
    excluded = {
        row["repository_identity"]
        for rows in excluded_plan["selected_repositories"].values()
        for row in rows
    }
    split = contract["assigned_split"]
    required_stratum = contract["required_path_stratum"]
    candidates = []
    for identity, metadata_row in metadata["repositories"].items():
        if identity in excluded or metadata_row["assigned_split"] != split:
            continue
        eligible_pulls = [
            row
            for row in metadata_row["pull_requests"]
            if row["path_metadata_complete"] is True and row["path_stratum"] == required_stratum
        ]
        if not eligible_pulls:
            continue
        representative = min(eligible_pulls, key=lambda row: int(row["number"]))
        source = dict(manifest["repositories"][identity])
        source["sampled_prs"] = [
            row for row in source["sampled_prs"] if int(row["number"]) == int(representative["number"])
        ]
        source["path_stratum"] = required_stratum
        source["path_metadata_complete"] = True
        candidates.append(source)
    candidates.sort(key=lambda row: (int(row["tree_path_count"]), row["repository_identity"]))
    count = int(contract["repository_count"])
    blockers = []
    selected = []
    if len(candidates) < count:
        blockers.append("insufficient_untouched_confirmatory_code_and_test_candidates")
    else:
        selected = [candidates[index] for index in _quantile_indices(len(candidates), count)]
    status = "frozen_before_tranche_content_fetch" if not blockers else contract["failure_decision"]
    report = {
        "schema_version": "temporal-code-confirmatory-execution-expansion-plan-v1",
        "status": status,
        "contract": contract,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(manifest_path): sha256_file(manifest_path),
            str(path_metadata_path): sha256_file(path_metadata_path),
            str(excluded_plan_path): sha256_file(excluded_plan_path),
        },
        "summary": {
            "candidate_count": len(candidates),
            "repository_count": len(selected),
            "blockers": blockers,
        },
        "content_fetch_limits": manifest["freeze_contract"]["content_fetch_limits"],
        "selected_repositories": {split: selected},
        "utility_scope": contract["utility_scope"],
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze confirmatory executable expansion.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--path-metadata", type=Path, default=DEFAULT_PATH_METADATA)
    parser.add_argument("--excluded-plan", type=Path, default=DEFAULT_EXCLUDED_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.manifest, args.path_metadata, args.excluded_plan, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0 if report["status"] == "frozen_before_tranche_content_fetch" else 2


if __name__ == "__main__":
    raise SystemExit(main())
