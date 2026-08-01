#!/usr/bin/env python3
"""Validate the code-domain v2 candidate-pool readiness report."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


REPORT = ROOT / "outputs" / "validation" / "code_domain_v2_candidate_pool_readiness_report.json"
DESIGN = ROOT / "configs" / "code_domain_next_development_cycle_v2_design.json"
DOC = ROOT / "docs" / "code_domain_v2_candidate_pool_readiness.md"


def main() -> int:
    report = load_json(REPORT)
    design = load_json(DESIGN)

    assert report["schema_version"] == "code-domain-v2-candidate-pool-readiness-v1"
    assert report["locked_prior_result"]["v1_status"] == "confirmatory_decision_reject_primary_margin_failure"
    assert report["locked_prior_result"]["v1_result_can_only_inform_separate_cycle"] is True
    assert report["requirements"]["minimum_stage_a_pass_repositories"] == design["candidate_pool_requirements"]["minimum_stage_a_pass_repositories"]
    assert report["requirements"]["insufficient_data_action"] == "insufficient_usable_data"

    for split in ("train", "development", "confirmatory"):
        profile = report["split_profiles"][split]
        assert profile["records"] >= 0
        assert profile["repository_count"] >= 0
        assert profile["token_proxy_sum"] >= 0
        assert 0.0 <= profile["largest_repository_token_share"] <= 1.0
        assert 0.0 <= profile["test_record_ratio"] <= 1.0

    assert report["requirement_checks"]["repository_disjointness"]["repository_disjoint"] is True
    assert report["requirement_checks"]["base_nll_scale"]["status"] == "required_later_before_development_promotion"
    assert report["requirement_checks"]["base_nll_scale"]["selector_use"] == "forbidden"

    forbidden = set(report["selector_signal_policy"]["forbidden_stage_b_signals"])
    for signal in ("Utility", "benchmark outcomes", "retention outcomes", "confirmatory model outcomes"):
        assert signal in forbidden
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert report["confirmatory_outcomes_read_for_v2"] is False
    assert "No Stage-B, Stage-C, Utility" in report["claim_boundary"]
    assert DOC.exists()
    doc_text = DOC.read_text(encoding="utf-8")
    assert "Code-Domain v2 Candidate-Pool Readiness" in doc_text
    assert "Utility remains Stage C" in doc_text
    print("[code-domain-v2-candidate-pool-readiness] corpus-shape contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
