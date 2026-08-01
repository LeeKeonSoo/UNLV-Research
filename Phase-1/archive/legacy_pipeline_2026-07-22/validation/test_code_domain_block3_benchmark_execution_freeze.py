#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    freeze = load_json(PROJECT_DIR / "configs" / "code_domain_block3_benchmark_execution_freeze_v1.json")

    assert freeze["status"] == "block3_code_benchmark_execution_frozen"
    assert freeze["decision"] == "run_lightweight_evalplus_generation_first_swebench_after_feasibility_gate"
    assert "Stage-C validation evidence only" in freeze["utility_scope"]

    contract = freeze["training_contract"]
    assert contract["base_model"] == "Qwen/Qwen3-4B-Base"
    assert contract["common_packed_token_budget"] == 325632
    assert contract["training_token_budget_cap"] == 327222
    assert contract["same_training_recipe_and_seed_set_required"] is True

    tiers = {tier["tier"]: tier for tier in freeze["benchmark_tiers"]}
    assert set(tiers) == {
        "tier0_completed_stage_c_support",
        "tier1_lightweight_code_generation",
        "tier2_swebench_capstone",
    }

    tier1 = tiers["tier1_lightweight_code_generation"]
    assert tier1["execution_decision"] == "execute_first"
    assert {"HumanEval+", "MBPP+"}.issubset(set(tier1["benchmarks"]))
    assert tier1["generation_script"] == "143_generate_code_domain_evalplus_samples.py"
    assert tier1["evaluation_script"] == "144_run_code_domain_evalplus_guardrail.py"
    assert tier1["success_rule"]["required_absolute_macro_pass_rate_improvement"] == 0.01
    assert tier1["selector_access"] is False

    commands = " ".join(tier1["command_templates"])
    assert "143_generate_code_domain_evalplus_samples.py missing" in commands
    assert "144_run_code_domain_evalplus_guardrail.py" in commands
    assert "HumanEval+,MBPP+" in commands

    tier2 = tiers["tier2_swebench_capstone"]
    assert tier2["execution_decision"] == "defer_until_feasibility_gate_passes"
    assert "SWE-bench Lite" in tier2["benchmarks"]
    assert "SWE-bench Verified" in tier2["benchmarks"]
    assert len(tier2["required_feasibility_gate"]) >= 5
    assert tier2["selector_access"] is False

    forbidden = " ".join(freeze["forbidden_uses"])
    assert "inside Stage-B selector objectives" in forbidden
    assert "benchmark task content as candidate training data" in forbidden

    readiness = PROJECT_DIR / freeze["readiness_source"]
    assert readiness.exists()

    print("[code-domain-block3-benchmark-execution-freeze] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
