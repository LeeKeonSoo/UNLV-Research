#!/usr/bin/env python3
"""Regression checks for the frozen temporal-code Stage-B ablation protocol."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_stage_b_ablation_protocol_v1.json")
    assert protocol["schema_version"] == "temporal-code-stage-b-ablation-protocol-v1"
    assert protocol["status"] == "frozen_before_target_model_development_results"
    arms = protocol["arms"]
    assert {
        "full_selector",
        "quality_only",
        "redundancy_only",
        "no_coverage_support",
        "stageA_random_equal_token",
        "raw_random_equal_token",
    }.issubset(arms)
    assert arms["full_selector"]["quality_weight"] == 0.8
    assert arms["full_selector"]["redundancy_support_weight"] == 0.2
    assert arms["quality_only"]["redundancy_support_weight"] == 0.0
    assert arms["redundancy_only"]["quality_weight"] == 0.0
    assert arms["no_coverage_support"]["coverage_support"] == "disabled"
    forbidden = protocol["shared_contract"]["forbidden_selector_signals"]
    assert "human or LLM review labels" in forbidden
    assert "Utility" in forbidden
    assert "confirmatory outcomes" in forbidden
    assert protocol["confirmatory_rule"]["policy_weights_thresholds_and_coverage_constraints_frozen"] is True
    assert protocol["confirmatory_rule"]["failure_action"].startswith("report negative finding or abstain")
    assert protocol["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-stage-b-ablation] frozen arms, equal-budget contract, and no-leak rule: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
