#!/usr/bin/env python3
"""Regression tests for retention-replay development contracts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    plans = [
        PROJECT_DIR / "configs" / "retention_replay_development_plan_qwen25_0p5b_fineweb.json",
        PROJECT_DIR / "configs" / "retention_replay_refinement_plan_qwen25_0p5b_fineweb.json",
        PROJECT_DIR / "configs" / "retention_replay_boundary_plan_qwen25_0p5b_fineweb.json",
    ]
    all_fractions = []
    for path in plans:
        plan = load_json(path)
        assert plan["schema_version"] == "retention-replay-development-plan-v1"
        assert plan["matched_comparator"] == "stageA_random_equal_budget"
        assert plan["utility_scope"] == "Stage C validation only; never selector objective"
        fractions = [float(value) for value in plan["candidate_target_fractions"]]
        assert all(0.0 < value <= 1.0 for value in fractions)
        assert len(fractions) == len(set(fractions))
        all_fractions.extend(fractions)
    assert 1.0 in all_fractions
    assert 0.99 in all_fractions
    recipe_plan = load_json(PROJECT_DIR / "configs" / "retention_recipe_development_plan_qwen25_0p5b_fineweb.json")
    assert recipe_plan["schema_version"] == "retention-recipe-development-plan-v1"
    assert recipe_plan["utility_scope"] == "Stage C validation only; never selector objective"
    assert set(recipe_plan["arms"]) == {
        "stageA_random_equal_budget",
        "retention_replay_target100",
        "retention_replay_target099",
    }
    frozen_plan_path = PROJECT_DIR / "configs" / "retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json"
    if frozen_plan_path.exists():
        frozen = load_json(frozen_plan_path)
        assert frozen["candidate"]["general_replay_fraction"] == 0.01
        assert frozen["candidate"]["training_recipe"]["learning_rate"] == 0.000005
        assert frozen["overall_success_rule"] == "Both fresh seeds must satisfy the per-seed joint rule."
        assert frozen["framework_scope"]["stage_b"] == "unchanged; no Utility or target-model objective"
        if "confirmatory_holdouts" in frozen:
            assert frozen["confirmatory_holdouts"]["status"] == "bound_before_confirmatory_training"
    trajectory = load_json(PROJECT_DIR / "configs" / "retention_training_trajectory_plan_qwen25_0p5b_fineweb.json")
    assert trajectory["schema_version"] == "retention-training-trajectory-plan-v1"
    assert trajectory["arms"] == ["stageA_random_equal_budget", "retention_replay_target099"]
    assert trajectory["utility_scope"] == "Stage C validation only; never selector objective"
    assert "tuning on frozen retention confirmatory holdouts" in trajectory["forbidden_uses"]
    assert trajectory["checkpoint_steps"] == sorted(set(trajectory["checkpoint_steps"]))
    release_protocol = load_json(PROJECT_DIR / "configs" / "target_effect_release_protocol_v1.json")
    assert release_protocol["schema_version"] == "target-effect-release-protocol-v1"
    assert release_protocol["non_retroactive"] is True
    assert release_protocol["confirmatory_requirements"]["minimum_fresh_training_seeds"] >= 5
    assert release_protocol["confirmatory_requirements"]["minimum_untouched_target_holdouts"] >= 2
    assert release_protocol["confirmatory_requirements"]["minimum_predeclared_task_suites"] >= 1
    assert release_protocol["framework_scope"]["stage_b"] == "unchanged chunk-level selection"
    assert release_protocol["utility_scope"] == "Stage C validation only; never selector objective"
    print("[retention-replay-contract] plans preserve Stage-C-only Utility and matched comparator: pass")
    print("[retention-replay-contract] recipe matrix fixes arms and preserves Stage-C-only Utility: pass")
    print("[retention-replay-contract] trajectory diagnostic is development-only: pass")
    print("[retention-replay-contract] future release protocol is non-retroactive and Stage-C-only: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
