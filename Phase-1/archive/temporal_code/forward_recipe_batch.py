#!/usr/bin/env python3
"""Freeze an outcome-independent execution-recipe batch from the candidate ledger."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_forward_e2_acquisition_v1.json"
DEFAULT_LEDGER = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_candidate_ledger.json"
DEFAULT_NATIVE_CONTRACT = Path("configs") / "temporal_code_native_execution_recipe_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_recipe_batch.json"


def freeze(
    contract_path: Path,
    ledger_path: Path,
    native_contract_path: Path,
    output_path: Path,
    start_index: int,
    client: Any,
    tree_client: Any,
    excluded_repositories: set[str] | None = None,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    ledger = load_json(ledger_path)
    native_contract = load_json(native_contract_path)
    native = importlib.import_module("98_freeze_temporal_code_native_execution_recipes")
    enrichment = importlib.import_module("65_enrich_temporal_code_repositories")
    maximum = int(contract["development_operations_contract"]["execution_batch_limit"])
    excluded = excluded_repositories or set()
    eligible = [row for row in ledger["candidates"] if row["repository_identity"] not in excluded]
    if start_index >= len(eligible):
        raise ValueError("No unprocessed candidate rows exist for the requested recipe batch.")
    recipes = {}
    skipped = []
    candidates = []
    for row in eligible[start_index:]:
        if len(recipes) >= maximum:
            break
        candidates.append(row)
        repository = row["repository_identity"]
        try:
            paths = tree_client.get_tree_paths(repository, row["merge_commit"])
        except RuntimeError as exc:
            skipped.append(
                {
                    "repository_identity": repository,
                    "pull_request_number": row["pull_request_number"],
                    "merge_commit": row["merge_commit"],
                    "reason": "tree_metadata_unavailable",
                    "error": str(exc),
                }
            )
            continue
        markers = enrichment._tree_evidence(paths)["python_project_marker_samples"]
        recipe = native._recipe(
            repository,
            list(markers),
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
            "assigned_split": "development",
            "evaluation_authorized_pending_e2_and_quarantine": False,
        }
    report = {
        "schema_version": "temporal-code-forward-development-recipe-batch-v1",
        "status": "forward_development_recipe_batch_frozen_before_execution",
        "contract": contract,
        "batch": {
            "start_index_within_unexcluded_candidates": start_index,
            "end_index_exclusive_within_unexcluded_candidates": start_index + len(candidates),
            "excluded_repository_count": len(excluded),
            "metadata_unavailable_skipped_count": len(skipped),
        },
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(ledger_path): sha256_file(ledger_path),
            str(native_contract_path): sha256_file(native_contract_path),
        },
        "repository_recipes": recipes,
        "summary": {
            "candidate_ledger_count": ledger["summary"]["candidate_count"],
            "unexcluded_candidate_count": len(eligible),
            "candidate_count": len(candidates),
            "execution_candidate_count": len(recipes),
            "metadata_unavailable_skipped_count": len(skipped),
            "project_tree_api_requests": tree_client.requests,
            "project_metadata_api_requests": client.requests,
        },
        "metadata_unavailable_repositories": [row["repository_identity"] for row in skipped],
        "skipped_candidates": skipped,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Development recipe batch only; no E2, Utility, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a forward development recipe batch.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--native-contract", type=Path, default=DEFAULT_NATIVE_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--exclude-recipe-dir", type=Path)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()
    native = importlib.import_module("98_freeze_temporal_code_native_execution_recipes")
    enrichment = importlib.import_module("65_enrich_temporal_code_repositories")
    token = native._resolve_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    excluded = set()
    if args.exclude_recipe_dir:
        for path in sorted(args.exclude_recipe_dir.glob("*.json")):
            payload = load_json(path)
            excluded.update((payload.get("repository_recipes") or {}).keys())
            excluded.update(str(value) for value in payload.get("metadata_unavailable_repositories") or [])
            excluded.update(
                str(row.get("repository_identity"))
                for row in payload.get("skipped_candidates") or []
                if row.get("repository_identity")
            )
    report = freeze(
        args.contract,
        args.ledger,
        args.native_contract,
        args.output,
        max(0, args.start_index),
        native.GitHubMetadataClient(token),
        enrichment.GitHubEnrichmentClient(token, delay_seconds=max(0.0, args.delay_seconds)),
        excluded,
    )
    print({"status": report["status"], "batch": report["batch"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
