#!/usr/bin/env python3
"""Freeze metadata-derived execution recipes for forward E2 pilot candidates."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_CANDIDATES = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_pilot_candidates.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_broad499.json"
DEFAULT_NATIVE_CONTRACT = Path("configs") / "temporal_code_native_execution_recipe_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_e2_pilot_recipes.json"


def freeze(
    contract_path: Path,
    candidates_path: Path,
    enrichment_path: Path,
    native_contract_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    forward = load_json(contract_path)
    candidates = load_json(candidates_path)
    enrichment = load_json(enrichment_path)
    native_contract = load_json(native_contract_path)
    native = importlib.import_module("98_freeze_temporal_code_native_execution_recipes")
    token = native._resolve_token()
    if not token:
        raise RuntimeError("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    client = native.GitHubMetadataClient(token)
    maximum = int(forward["infrastructure_pilot"]["maximum_execution_candidates"])
    selected = candidates["candidates"][:maximum]
    recipes = {}
    for row in selected:
        repository = row["repository_identity"]
        markers = list(enrichment["repositories"][repository]["tree_evidence"]["python_project_marker_samples"])
        recipe = native._recipe(
            repository,
            markers,
            list(row["changed_test_paths"]),
            row["merge_commit"],
            client,
            native_contract,
        )
        recipes[repository] = {
            **recipe,
            "pull_request_number": row["pull_request_number"],
            "parent_commit": row["parent_commit"],
            "merge_commit": row["merge_commit"],
            "merge_timestamp": row["merge_timestamp"],
            "repository_url": row["repository_url"],
            "pilot_task_evaluation_authorized": False,
        }
    report = {
        "schema_version": "temporal-code-forward-e2-pilot-recipes-v1",
        "status": "frozen_before_forward_pilot_execution",
        "contract": forward,
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(candidates_path): sha256_file(candidates_path),
            str(enrichment_path): sha256_file(enrichment_path),
            str(native_contract_path): sha256_file(native_contract_path),
        },
        "repository_recipes": recipes,
        "summary": {
            "candidate_count": len(candidates["candidates"]),
            "execution_candidate_count": len(recipes),
            "github_api_requests": client.requests,
        },
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "utility_scope": forward["utility_scope"],
        "claim_boundary": forward["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze forward E2 pilot recipes.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--native-contract", type=Path, default=DEFAULT_NATIVE_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = freeze(args.contract, args.candidates, args.enrichment, args.native_contract, args.output)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
