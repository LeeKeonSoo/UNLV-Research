#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


PROTOCOL_PATH = PROJECT_DIR / "configs" / "math_domain_stage_c_protocol_qwen3_4b_v1.json"


def main() -> int:
    protocol = load_json(PROTOCOL_PATH)
    assert protocol["status"] == "frozen_before_math_stage_c_training_or_benchmark_outcomes"
    assert protocol["stage_c_outcomes_read"] is False
    assert "Stage C validation only" in protocol["utility_scope"]

    arms = set(protocol["training_arms"])
    assert {
        "base_no_update",
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_math_equal_budget",
        "known_high_quality_equal_budget",
    }.issubset(arms)

    recipe = protocol["confirmatory_training_recipe"]
    assert recipe["training_token_budget_cap"] == 119163
    assert recipe["confirmatory_training_seeds"] == [101, 131, 163]

    heldout = protocol["heldout_nll"]["frozen_heldout"]
    assert Path(heldout["path"]).exists()
    assert heldout["selected_token_proxy"] >= heldout["token_proxy_budget"]
    assert heldout["candidate_records"] > heldout["selected_records"] > 0

    benchmarks = {item["name"]: item for item in protocol["stage_c_benchmarks"]}
    assert benchmarks["GSM8K"]["dataset_id"] == "openai/gsm8k"
    assert benchmarks["MATH"]["dataset_id"] == "hendrycks/competition_math"

    forbidden = " ".join(protocol["forbidden_uses"])
    assert "outcomes in Stage B" in forbidden
    assert "benchmark content as training candidates" in forbidden

    commands = " ".join(protocol["command_templates"])
    assert "prepare-blocks" in commands
    assert "train-missing" in commands
    assert "eval-missing" in commands
    assert "GSM8K,MATH" in commands

    print("[math-domain-stage-c-protocol] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
