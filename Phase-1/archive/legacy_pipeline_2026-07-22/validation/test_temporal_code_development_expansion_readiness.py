#!/usr/bin/env python3
"""Contract checks for development executable-expansion readiness."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_development_expansion_readiness.json"
    )
    assert report["status"] == "development_stage_c_blocked_insufficient_executable_holdout"
    assert report["summary"]["frozen_candidate_repositories"] == 11
    assert report["summary"]["collection_gate_pass_bundles"] == 7
    assert report["summary"]["generic_execution_verified_bundles"] == 0
    assert report["summary"]["total_verified_development_bundles"] == 1
    assert report["decision"]["development_utility_may_start"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    forbidden = set(report["decision"]["forbidden_reactions"])
    assert "tune Stage B from execution or future Utility outcomes" in forbidden
    print("[temporal-code-development-expansion-readiness] insufficient executable holdout abstention: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
