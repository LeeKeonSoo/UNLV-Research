#!/usr/bin/env python3
"""Validate code-domain v2 expansion collection artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


PLAN = ROOT / "outputs" / "temporal_code_collection" / "code_domain_v2_expansion_tranche_plan.json"
COMBINED_STAGE0 = ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "stage0_combined_report.json"
READINESS = ROOT / "outputs" / "validation" / "code_domain_v2_candidate_pool_readiness_report.json"


def main() -> int:
    plan = load_json(PLAN)
    assert plan["schema_version"] == "code-domain-v2-expansion-tranche-plan-v1"
    assert plan["status"] == "frozen_before_expansion_content_fetch"
    assert plan["summary"]["repository_count"] == plan["summary"]["maximum_bundle_count"]
    assert not plan["summary"]["blockers"]
    forbidden = set(plan["contract"]["selection_forbids"])
    for signal in ("Stage-A outcomes", "Stage-B outcomes", "Utility", "benchmark outcomes", "confirmatory model outcomes"):
        assert signal in forbidden
    assert plan["utility_scope"] == "Stage C validation only; never selector objective"

    if COMBINED_STAGE0.exists():
        combined = load_json(COMBINED_STAGE0)
        assert combined["status"] == "stage0_pools_merged_before_stage_a"
        assert combined["summary"]["duplicate_record_id_count"] == 0
        assert combined["utility_scope"] == "Stage C validation only; never selector objective"

    if READINESS.exists():
        readiness = load_json(READINESS)
        assert readiness["locked_prior_result"]["v1_status"] == "confirmatory_decision_reject_primary_margin_failure"
        assert readiness["utility_scope"] == "Stage C validation only; never selector objective"

    print("[code-domain-v2-expansion] expansion collection contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
