#!/usr/bin/env python3
"""Validate frozen code-domain v2 Stage-B arms."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


REPORT = ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "stage_b_v2_arms_report.json"
DESIGN = ROOT / "configs" / "code_domain_next_development_cycle_v2_design.json"
OUTPUT_DIR = ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2"


def main() -> int:
    report = load_json(REPORT)
    design = load_json(DESIGN)
    assert report["status"] == "stage_b_v2_arms_frozen_before_stage_c"
    assert report["inputs"]["stage_a_dir"].endswith("stage_a_code_domain_v2_balanced")
    assert report["inputs"]["train_stage_a_pass_chunks"] > 0
    assert report["disjointness"]["curated_v2_stageA_random_disjoint"] is True
    assert report["disjointness"]["intersection_count"] == 0

    expected = set(design["stage_b_v2_proxy_plan"]["required_ablations"])
    assert expected.issubset(set(report["required_ablations_from_design"]))
    for arm in [
        "full_selector",
        "quality_only",
        "redundancy_only",
        "no_coverage_support",
        "no_test_code_balance",
        "no_repository_diversity_cap",
    ]:
        assert arm in report["ablations"]
        assert report["ablations"][arm]["selected_chunks"] > 0

    for path_name in [
        "curated_v2_equal_budget.jsonl",
        "stageA_random_equal_budget.jsonl",
        "raw_random_equal_budget.jsonl",
        "known_high_quality_equal_budget.jsonl",
        "quality_only_selected.jsonl",
        "redundancy_only_selected.jsonl",
        "no_coverage_support_selected.jsonl",
        "no_test_code_balance_selected.jsonl",
        "no_repository_diversity_cap_selected.jsonl",
    ]:
        assert (OUTPUT_DIR / path_name).exists()

    curated = report["primary_arms"]["curated_v2_equal_budget"]["token_proxy_count"]
    stage_a = report["primary_arms"]["stageA_random_equal_budget"]["token_proxy_count"]
    raw = report["primary_arms"]["raw_random_equal_budget"]["token_proxy_count"]
    known_hq = report["primary_arms"]["known_high_quality_equal_budget"]["token_proxy_count"]
    assert stage_a <= curated
    assert raw >= curated
    assert known_hq >= curated

    forbidden = set(report["selection_forbids"])
    for signal in ("Utility", "benchmark outcomes", "retention outcomes", "confirmatory model outcomes"):
        assert signal in forbidden
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert report["confirmatory_outcomes_read_for_v2"] is False
    assert "Stage-B v2 arm freeze only" in report["claim_boundary"]
    print("[code-domain-v2-stage-b] frozen arms and ablations: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
