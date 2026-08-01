#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    contract = load_json(ROOT / "configs" / "raw_corpus_matrix_stage_c_study_v1.json")
    assert contract["status"] == "frozen_before_stage_c_model_outcomes"
    assert contract["target_model"]["model_id"] == "Qwen/Qwen3-4B-Base"
    assert contract["primary_study"]["condition"] == "raw_mixed"
    assert contract["primary_study"]["comparison"] == "curated_equal_token_vs_stage_a_random_equal_token"
    assert contract["condition_roles"]["clean_retain_all"] == "retain_all_policy_check"
    assert contract["condition_roles"]["risk_heavy"] == "stress_sensitivity"
    assert contract["stage_b_isolation"]["utility_available_to_stage_b"] is False
    assert contract["stage_b_isolation"]["source_tier_available_to_stage_b"] is False
    assert contract["holdout_contract"]["development"]["stage_b_read"] is False
    assert contract["holdout_contract"]["confirmatory"]["stage_b_read"] is False
    assert contract["holdout_contract"]["confirmatory"]["outcomes_read_before_development_decision"] is False
    assert contract["seed_contract"]["development_training_seeds"] == [11, 23, 37, 53, 71]
    assert contract["seed_contract"]["confirmatory_training_seeds"] == [101, 131, 163, 197, 239]
    assert contract["natural_budget"]["role"] == "supporting_cost_performance_tradeoff"
    print("[raw-corpus-matrix-stage-c-study] primary, holdout, and isolation contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
