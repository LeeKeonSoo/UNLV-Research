#!/usr/bin/env python3
"""Merge frozen metadata-only forward repository discovery strata."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.temporal_code_manifests import build_repository_split_manifest


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_INPUTS = [
    OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_expanded.json",
    OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_stars_5_19.json",
    OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_stars_0_4.json",
]
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "forward_development_repository_discovery_combined.json"


def merge(protocol_path: Path, contract_path: Path, input_paths: List[Path], output_path: Path) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    contract = load_json(contract_path)
    manifests = [load_json(path) for path in input_paths]
    candidates: Dict[str, Dict[str, Any]] = {}
    query_sets = []
    for manifest in manifests:
        query_sets.append(manifest["queries"])
        for identity, row in manifest["candidates"].items():
            candidates[identity] = row
    eligible = [row for row in candidates.values() if row["eligible_for_metadata_enrichment"]]
    split = build_repository_split_manifest(eligible, protocol)
    report = {
        "schema_version": "temporal-code-forward-repository-discovery-combined-v1",
        "status": "frozen_combined_repository_frame_before_task_metadata",
        "protocol_name": protocol["protocol_name"],
        "contract_status": contract["development_source_coverage_amendment"]["status"],
        "metadata_only": True,
        "query_sets": query_sets,
        "source_sha256": {
            str(protocol_path): sha256_file(protocol_path),
            str(contract_path): sha256_file(contract_path),
            **{str(path): sha256_file(path) for path in input_paths},
        },
        "summary": {
            "source_manifest_count": len(manifests),
            "candidate_count": len(candidates),
            "metadata_enrichment_candidate_count": len(eligible),
            "excluded_repository_count": len(candidates) - len(eligible),
            "preliminary_split_counts": split["split_counts"],
        },
        "candidates": dict(sorted(candidates.items())),
        "preliminary_split_manifest": split,
        "task_metadata_read": False,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Combined repository discovery frame only; no task, E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge forward repository discovery strata.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = merge(args.protocol, args.contract, args.inputs, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
