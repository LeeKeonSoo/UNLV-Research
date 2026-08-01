#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_eval_common import load_json, sha256_file  # noqa: E402


def main() -> int:
    plan_path = ROOT / "configs" / "raw_corpus_matrix_development_execution_qwen3_4b_v1.json"
    plan = load_json(plan_path)
    assert plan["status"] == "frozen_before_raw_matrix_development_outcomes"
    assert plan["training_arms"] == [
        "raw_mixed_random_equal_token",
        "stage_a_random_equal_token",
        "curated_equal_token",
    ]
    assert plan["training_recipe"]["optimizer_steps"] == 27
    assert plan["training_recipe"]["development_training_seeds"] == [11, 23, 37, 53, 71]
    assert plan["training_recipe"]["common_packed_token_budget"] == 452608
    source_contract = plan["source_study_contract"]
    assert sha256_file(ROOT / source_contract["path"]) == source_contract["sha256"]
    for arm, entry in plan["training_blocks"]["blocks"].items():
        assert sha256_file(ROOT / entry["path"]) == entry["sha256"], arm
    heldout = plan["heldout_nll"]["frozen_heldout"]
    assert sha256_file(ROOT / heldout["path"]) == heldout["sha256"]
    assert plan["stage_b_isolation"]["utility_available_to_stage_b"] is False
    assert plan["confirmatory_outcomes_read"] is False
    print("[raw-corpus-matrix-development-execution] frozen blocks, seeds, and holdout: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
