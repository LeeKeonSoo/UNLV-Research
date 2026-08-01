#!/usr/bin/env python3
"""Validate the code-domain next-cycle design contract."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


CONFIG = ROOT / "configs" / "code_domain_next_development_cycle_v2_design.json"
DOC = ROOT / "docs" / "code_domain_next_development_cycle_design.md"
POSTMORTEM = ROOT / "outputs" / "validation" / "code_domain_confirmatory_postmortem_report.json"


def main() -> int:
    config = load_json(CONFIG)
    postmortem = load_json(POSTMORTEM)

    assert config["status"] == "design_draft_not_executable_protocol"
    assert config["source_postmortem"]["locked_v1_status"] == postmortem["confirmatory_result"]["status"]
    assert postmortem["confirmatory_result"]["status"] == "confirmatory_decision_reject_primary_margin_failure"
    assert postmortem["decision_implications"]["primary_margin_passed"] is False

    assert config["stage_boundaries"]["stage_a"] == "chunk-level hard gate"
    assert config["stage_boundaries"]["stage_b"] == "chunk-level selection among Stage-A-pass chunks"
    assert config["stage_boundaries"]["stage_c"] == "subset/model validation only"
    assert config["stage_boundaries"]["utility_scope"] == "Stage C validation only; never selector objective"

    forbidden = set(config["selector_signal_policy"]["forbidden_stage_b_signals"])
    for signal in [
        "Utility",
        "benchmark outcomes",
        "retention outcomes",
        "development model outcomes",
        "confirmatory model outcomes",
        "human review labels",
        "LLM review labels",
    ]:
        assert signal in forbidden

    pool = config["candidate_pool_requirements"]
    assert pool["split_contract"] == "time-disjoint and repository-disjoint"
    assert pool["minimum_stage_a_pass_repositories"]["train"] >= 30
    assert pool["minimum_stage_a_pass_repositories"]["development_heldout"] >= 10
    assert pool["minimum_stage_a_pass_repositories"]["confirmatory_heldout"] >= 10
    assert pool["maximum_token_share_per_repository"] <= 0.25
    assert pool["maximum_development_confirmatory_test_ratio_difference"] <= 0.05
    assert pool["insufficient_data_action"] == "insufficient_usable_data"

    stage_b = config["stage_b_v2_proxy_plan"]
    assert "stageA_random_equal_budget" in stage_b["required_ablations"]
    assert "raw_random_equal_budget" in stage_b["required_ablations"]
    assert "no_test_code_balance" in stage_b["required_ablations"]
    assert "no_repository_diversity_cap" in stage_b["required_ablations"]

    margin = config["margin_calibration"]
    assert margin["status"] == "development_only_calibration_required_before_v2_confirmatory_freeze"
    assert "v2 confirmatory outcomes" in margin["inputs_forbidden"]
    assert "choose exactly one primary success rule before v2 confirmatory outcomes are read" in margin["freeze_requirement"]

    assert "common_disjoint_stageA_baseline_for_sensitivity_arms" in config["confirmatory_freeze_requirements"]
    assert config["confirmatory_outcomes_read_for_v2"] is False
    assert "Design draft only" in config["claim_boundary"]

    doc_text = DOC.read_text(encoding="utf-8")
    assert "does not revise, rescue, or reinterpret" in doc_text
    assert "Stage C remains subset/model validation" in doc_text
    assert "Utility" in doc_text and "Stage-B selector objective" in doc_text
    assert "insufficient_usable_data" in doc_text
    print("[code-domain-next-cycle-design] v2 design boundaries and calibration contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
