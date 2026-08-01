#!/usr/bin/env python3
"""Contract checks for no-outcome development executable expansion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    collection = PROJECT_DIR / "outputs" / "temporal_code_collection"
    expansion = load_json(collection / "temporal_code_development_execution_expansion_plan.json")
    primary = load_json(collection / "temporal_code_path_stratified_tranche_plan_v2.json")
    expansion_rows = expansion["selected_repositories"]["development"]
    expansion_ids = {row["repository_identity"] for row in expansion_rows}
    primary_ids = {
        row["repository_identity"] for rows in primary["selected_repositories"].values() for row in rows
    }
    assert expansion["status"] == "frozen_before_tranche_content_fetch"
    assert len(expansion_rows) >= 11
    assert not expansion_ids.intersection(primary_ids)
    assert all(row["assigned_split"] == "development" for row in expansion_rows)
    assert all(len(row["sampled_prs"]) == 1 for row in expansion_rows)
    assert expansion["summary"]["development_utility_remains_blocked"] is True
    forbidden = set(expansion["contract"]["selection_forbids"])
    assert {"test execution outcomes", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
        forbidden
    )
    assert expansion["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-development-expansion] no-outcome all-eligible selection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
