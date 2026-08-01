#!/usr/bin/env python3
"""Contract checks for the fresh metadata-only development expansion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    collection = PROJECT_DIR / "outputs" / "temporal_code_collection"
    fresh = load_json(collection / "temporal_code_development_fresh_expansion_plan.json")
    prior_paths = [
        collection / "temporal_code_path_stratified_tranche_plan_v2.json",
        collection / "temporal_code_development_execution_expansion_plan.json",
    ]
    prior_ids = set()
    for path in prior_paths:
        plan = load_json(path)
        prior_ids.update(
            row["repository_identity"]
            for rows in plan["selected_repositories"].values()
            for row in rows
        )
    rows = fresh["selected_repositories"]["development"]
    ids = {row["repository_identity"] for row in rows}
    assert fresh["status"] == "frozen_before_tranche_content_fetch"
    assert len(rows) == 14
    assert not ids.intersection(prior_ids)
    assert all(row["assigned_split"] == "development" for row in rows)
    assert all(row["path_stratum"] in {"test_only", "code_only"} for row in rows)
    assert all(len(row["sampled_prs"]) == 1 for row in rows)
    assert fresh["summary"]["development_utility_remains_blocked"] is True
    forbidden = set(fresh["contract"]["selection_forbids"])
    assert {"generic execution outcomes", "native execution outcomes", "Utility", "benchmark outcomes"}.issubset(
        forbidden
    )
    assert fresh["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-development-fresh-expansion] unused metadata-only frame freeze: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
