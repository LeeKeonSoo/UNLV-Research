#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hard_quality_profile import build_hard_quality_candidate_plan


DECLARATION = {
    "model_id": "Qwen/Qwen3-4B-Base",
    "tokenizer_sha256": "fixture-tokenizer",
    "training_recipe_fingerprint": "fixture-recipe",
    "intended_use": "code-continued-pretraining",
    "max_training_tokens": 120,
}


def test_hard_quality_plan_uses_only_explicit_user_budget_and_frozen_mid_evidence() -> None:
    # Given: a declared token budget and three candidate-only Mid group summaries.
    groups = [
        {"group_id": "high-yield", "effect_estimate": 0.12, "upper_confidence_bound": 0.15, "token_count": 60},
        {"group_id": "low-yield", "effect_estimate": 0.03, "upper_confidence_bound": 0.04, "token_count": 60},
        {"group_id": "candidate-remove", "effect_estimate": -0.04, "upper_confidence_bound": -0.01, "token_count": 40},
    ]

    # When: an opt-in Hard candidate plan is created.
    plan = build_hard_quality_candidate_plan(groups=groups, declaration=DECLARATION)

    # Then: selection is deterministic, budget-bounded, and no implicit retention ratio exists.
    assert plan["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert plan["budget"]["declared_max_training_tokens"] == 120
    assert plan["budget"]["selected_token_count"] == 120
    assert [row["group_id"] for row in plan["selected_groups"]] == ["high-yield", "low-yield"]
    assert plan["excluded_groups"][0]["reason_code"] == "mid_quality_calibrated_non_positive_candidate"
    assert "retention_fraction" not in plan["budget"]


def test_hard_quality_plan_rejects_hidden_retention_fraction() -> None:
    # Given: a declaration that attempts to smuggle in a retention fraction.
    declaration = {**DECLARATION, "retention_fraction": 0.4}

    # When / Then: Hard planning refuses the hidden selector target.
    try:
        build_hard_quality_candidate_plan(
            groups=[{"group_id": "useful", "effect_estimate": 0.1, "upper_confidence_bound": 0.1, "token_count": 10}],
            declaration=declaration,
        )
    except RuntimeError as error:
        assert "retention" in str(error)
    else:
        raise AssertionError("Hard profile must reject an implicit retention fraction")


if __name__ == "__main__":
    test_hard_quality_plan_uses_only_explicit_user_budget_and_frozen_mid_evidence()
    test_hard_quality_plan_rejects_hidden_retention_fraction()
    print("[hard-quality-profile] explicit-budget candidate plan: pass")
