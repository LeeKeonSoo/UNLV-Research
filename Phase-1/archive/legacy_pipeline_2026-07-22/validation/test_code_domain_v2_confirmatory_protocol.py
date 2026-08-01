#!/usr/bin/env python3
"""Contract checks for the frozen code-domain v2 confirmatory protocol."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "code_domain_v2_confirmatory_protocol_qwen3_4b.json")
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "code_domain_v2_confirmatory_protocol_qwen3_4b_report.json"
    )
    development = load_json(PROJECT_DIR / "configs" / "code_domain_v2_development_plan_qwen3_4b.json")
    decision = load_json(
        PROJECT_DIR / "outputs" / "validation" / "code_domain_v2_development_decision_report.json"
    )
    retention = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
    )

    assert decision["status"] == "development_decision_promote_to_confirmatory"
    assert protocol["status"] == "frozen_before_v2_confirmatory_training_outcomes"
    assert report["status"] == "v2_confirmatory_protocol_frozen"
    assert protocol["confirmatory_outcomes_read"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert protocol["utility_scope"] == "Stage C validation only; never selector objective"
    assert "using different Stage-A baselines for sensitivity arms" in protocol["forbidden_uses"]
    assert (
        "using Utility, benchmark outcomes, retention outcomes, development outcomes, confirmatory outcomes, or human/LLM review labels in Stage B"
        in protocol["forbidden_uses"]
    )

    seeds = protocol["confirmatory_training_recipe"]["confirmatory_training_seeds"]
    dev_seeds = development["training_recipe"]["development_training_seeds"]
    retention_seeds = retention["contract"]["seed_contract"]["confirmatory_training_seeds"]
    assert seeds == retention_seeds
    assert len(seeds) == 5
    assert not set(seeds).intersection(set(dev_seeds))
    assert protocol["confirmatory_training_recipe"]["same_seed_set_for_every_arm"] is True

    for key in (
        "optimizer_steps",
        "gradient_accumulation_steps",
        "common_packed_token_budget",
        "training_token_budget_cap",
        "sequence_length",
        "learning_rate",
        "weight_decay",
    ):
        assert protocol["confirmatory_training_recipe"][key] == development["training_recipe"][key], key

    assert protocol["primary_comparison"] == development["primary_comparison"]
    assert protocol["training_arms"] == development["training_arms"]
    assert protocol["source_stage_b_arms"]["curated_v2_stageA_random_disjoint"] is True

    rule = protocol["primary_success_rule"]
    assert rule["all_conditions_required"] is True
    assert rule["primary_treatment"] == "curated_v2_equal_budget"
    assert rule["primary_baseline"] == "stageA_random_equal_budget"
    assert rule["required_absolute_nll_reduction"] == 0.003
    assert "all curated_v2 seed-level NLLs" in rule["paired_seed_requirement"]

    calibration = protocol["margin_calibration"]
    assert calibration["confirmatory_outcomes_used"] is False
    assert calibration["frozen_absolute_nll_margin"] == 0.003
    assert calibration["development_primary_mean_delta"] > calibration["frozen_absolute_nll_margin"]
    assert calibration["development_primary_min_delta"] > calibration["frozen_absolute_nll_margin"]

    heldout = protocol["heldout_nll"]["frozen_heldout"]
    assert protocol["heldout_nll"]["source_split"] == "confirmatory"
    assert "stage_a_code_domain_v2_balanced" in protocol["heldout_nll"]["source_file"]
    assert heldout["source_split"] == "confirmatory"
    assert heldout["selected_records"] > 0
    assert heldout["selected_token_proxy"] <= heldout["token_proxy_budget"] + 4096
    assert heldout["repository_count"] >= 10
    assert Path(heldout["path"]).exists()

    guardrails = protocol["stage_c_guardrails"]
    assert guardrails["evalplus_confirmatory"]["required_split"] == "confirmatory"
    assert guardrails["evalplus_confirmatory"]["non_inferiority"]["maximum_allowed_absolute_regression_macro"] == 0.02
    assert guardrails["general_text_nll_retention"]["maximum_allowed_mean_nll_increase"] == 0.01
    assert guardrails["general_task_retention"]["maximum_allowed_absolute_regression_macro"] == 0.01
    assert guardrails["decision_rule"]["all_guardrails_mandatory"] is True
    assert guardrails["decision_rule"]["confirmatory_may_not_select_recipe"] is True

    for arm in protocol["training_arms"]:
        if arm == "base_no_update":
            continue
        block = protocol["training_blocks"]["blocks"][arm]
        payload = protocol["training_payloads"][arm]
        assert Path(block["path"]).exists()
        assert Path(payload["jsonl_path"]).exists()
        assert block["sha256"]
        assert payload["jsonl_sha256"]

    print("[code-domain-v2-confirmatory] frozen protocol preserves margin, seeds, and outcome isolation: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
