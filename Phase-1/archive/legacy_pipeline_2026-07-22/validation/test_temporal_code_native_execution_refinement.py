#!/usr/bin/env python3
"""Contract checks for native execution refinement stopping decision."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_native_execution_refinement_report.json"
    )
    summary = report["summary"]
    decision = report["decision"]
    assert report["status"] == "native_recipe_exploration_no_executable_recovery"
    assert summary["native_v1_build_pass_commits"] > summary["generic_build_pass_commits"]
    assert summary["native_v2_verified_bundles"] == 0
    assert decision["development_utility_may_start"] is False
    assert decision["continue_recipe_tuning_on_same_development_pool"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-native-refinement] no-recovery stopping boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
