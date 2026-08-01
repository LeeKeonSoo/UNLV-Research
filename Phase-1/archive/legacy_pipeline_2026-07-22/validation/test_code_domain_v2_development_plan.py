#!/usr/bin/env python3
"""Validate frozen code-domain v2 development plan."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


PLAN = ROOT / "configs" / "code_domain_v2_development_plan_qwen3_4b.json"
REPORT = ROOT / "outputs" / "validation" / "code_domain_v2_development_plan_qwen3_4b_report.json"
STAGE_B = ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "stage_b_v2_arms_report.json"


def main() -> int:
    plan = load_json(PLAN)
    report = load_json(REPORT)
    stage_b = load_json(STAGE_B)
    assert plan["status"] == "frozen_before_v2_development_training_outcomes"
    assert report["status"] == "v2_development_plan_frozen"
    assert plan["confirmatory_outcomes_read"] is False
    assert report["confirmatory_outcomes_read"] is False
    assert plan["utility_scope"] == "Stage C validation only; never selector objective"
    assert "V2 development-plan freeze only" in plan["claim_boundary"]

    assert plan["training_arms"] == [
        "base_no_update",
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_v2_equal_budget",
        "known_high_quality_equal_budget",
    ]
    assert plan["primary_comparison"]["treatment"] == "curated_v2_equal_budget"
    assert plan["primary_comparison"]["primary_baseline"] == "stageA_random_equal_budget"
    assert plan["training_recipe"]["same_seed_set_for_every_arm"] is True
    assert len(plan["training_recipe"]["development_training_seeds"]) == 5
    assert plan["training_recipe"]["optimizer_steps"] == 20
    assert plan["training_recipe"]["training_token_budget_cap"] == stage_b["primary_arms"]["curated_v2_equal_budget"]["token_proxy_count"]
    assert plan["training_recipe"]["common_packed_token_budget"] > 0

    blocks = report["summary"]["training_blocks"]
    assert blocks["status"] == "v2_development_training_blocks_frozen"
    packed = {row["packed_tokens"] for row in blocks["blocks"].values()}
    assert len(packed) == 1
    for arm in plan["training_arms"]:
        if arm == "base_no_update":
            continue
        block = blocks["blocks"][arm]
        assert Path(block["path"]).exists()
        assert block["packed_tokens"] == plan["training_recipe"]["common_packed_token_budget"]

    heldout = plan["heldout_nll"]["frozen_heldout"]
    assert heldout["source_split"] == "development"
    assert Path(heldout["path"]).exists()
    assert heldout["selected_records"] > 0
    assert heldout["repository_count"] >= 10
    assert set(heldout["allowed_content_types"]) == {"code", "test"}
    assert plan["heldout_nll"]["confirmatory_read_forbidden"] is True

    forbidden = " ".join(plan["forbidden_uses"])
    assert "Utility" in forbidden and "Stage B" in forbidden
    assert "confirmatory outcomes" in forbidden
    assert plan["margin_calibration"]["status"] == "development_only_calibration_required_before_v2_confirmatory_freeze"
    print("[code-domain-v2-development-plan] frozen heldout, blocks, and training contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
