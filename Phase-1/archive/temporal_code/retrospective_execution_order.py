#!/usr/bin/env python3
"""Freeze an outcome-independent execution order for remaining candidates."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_LEDGER = COLLECTION / "temporal_code_retrospective_combined_candidate_ledger.json"
DEFAULT_RECIPE_DIR = COLLECTION
DEFAULT_OUTPUT = COLLECTION / "temporal_code_retrospective_execution_order_ledger.json"
ORDER_SEED = "temporal-code-retrospective-execution-order-v1"


def build(ledger_path: Path, recipe_dir: Path, output_path: Path) -> Dict[str, Any]:
    ledger = load_json(ledger_path)
    attempted = set()
    recipe_paths = sorted(recipe_dir.glob("temporal_code_retrospective_recipe_batch_*.json"))
    for path in recipe_paths:
        attempted.update((load_json(path).get("repository_recipes") or {}).keys())
    remaining = [row for row in ledger["candidates"] if row["repository_identity"] not in attempted]
    remaining.sort(
        key=lambda row: (
            hashlib.sha256(f"{ORDER_SEED}:{row['repository_identity']}".encode()).hexdigest(),
            row["repository_identity"],
        )
    )
    report = {
        "schema_version": "temporal-code-retrospective-execution-order-ledger-v1",
        "status": "remaining_execution_order_frozen_before_further_e2_outcomes",
        "order_rule": "ascending sha256 of fixed seed plus repository identity",
        "order_seed": ORDER_SEED,
        "source_sha256": {
            str(ledger_path): sha256_file(ledger_path),
            **{str(path): sha256_file(path) for path in recipe_paths},
        },
        "summary": {
            "candidate_count": int(ledger["summary"]["candidate_count"]),
            "already_frozen_recipe_repository_count": len(attempted),
            "already_frozen_recipe_repository_in_ledger_count": sum(
                identity in {row["repository_identity"] for row in ledger["candidates"]} for identity in attempted
            ),
            "remaining_candidate_count": len(remaining),
        },
        "candidates": remaining,
        "execution_outcomes_read": False,
        "confirmatory_outcomes_read": False,
        "development_utility_may_start": False,
        "utility_scope": ledger["utility_scope"],
        "claim_boundary": "Remaining execution order only; no E2, Utility, selector, or curation claim.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze remaining retrospective execution order.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--recipe-dir", type=Path, default=DEFAULT_RECIPE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.ledger, args.recipe_dir, args.output)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
