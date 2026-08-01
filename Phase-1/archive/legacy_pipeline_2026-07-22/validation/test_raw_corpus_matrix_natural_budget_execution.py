#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_eval_common import load_json, sha256_file  # noqa: E402


def main() -> int:
    plan_path = ROOT / "configs" / "raw_corpus_matrix_natural_budget_execution_qwen3_4b_v2.json"
    plan = load_json(plan_path)
    assert plan["status"] == "frozen_before_natural_budget_development_outcomes"
    assert plan["training_arms"] == ["raw_mixed_all_natural", "curated_natural"]
    assert plan["primary_comparison"]["treatment"] == "curated_natural"
    assert plan["primary_comparison"]["primary_baseline"] == "raw_mixed_all_natural"
    assert plan["training_recipe"]["optimizer_steps_by_arm"] == {
        "raw_mixed_all_natural": 42,
        "curated_natural": 27,
    }
    for arm, entry in plan["training_blocks"]["blocks"].items():
        assert sha256_file(ROOT / entry["path"]) == entry["sha256"], arm
    heldout = plan["heldout_nll"]["frozen_heldout"]
    assert sha256_file(ROOT / heldout["path"]) == heldout["sha256"]
    assert plan["natural_budget_reporting"]["required_metrics"] == [
        "packed_training_tokens",
        "effective_training_tokens",
        "optimizer_steps",
        "elapsed_seconds",
        "development_code_heldout_mean_nll",
        "evalplus_humaneval_plus_mbpp_macro_pass_rate",
    ]
    assert plan["stage_b_isolation"]["utility_available_to_stage_b"] is False
    assert plan["prior_equal_token_outcomes_read"] is True
    print("[raw-corpus-matrix-natural-budget-execution] frozen arms, budgets, and holdout: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
