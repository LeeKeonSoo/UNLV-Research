#!/usr/bin/env python3
"""Validate the code-domain v2 balanced Stage-A pool."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


BALANCED = ROOT / "outputs" / "temporal_code_collection" / "stage_a_code_domain_v2_balanced" / "balanced_stage_a_pool_report.json"
READINESS = ROOT / "outputs" / "validation" / "code_domain_v2_candidate_pool_readiness_report.json"


def main() -> int:
    balanced = load_json(BALANCED)
    readiness = load_json(READINESS)
    assert balanced["status"] == "balanced_stage_a_pool_built_before_stage_b_or_stage_c"
    assert balanced["utility_scope"] == "Stage C validation only; never selector objective"
    forbidden = set(balanced["selection_forbids"])
    for signal in ("Utility", "benchmark outcomes", "development model outcomes", "confirmatory model outcomes"):
        assert signal in forbidden
    for split in ("train", "development", "confirmatory"):
        after = balanced["split_summaries"][split]["after"]
        assert after["repository_count"] >= 10 if split != "train" else after["repository_count"] >= 30
        assert after["largest_repository_token_share"] <= 0.25
    assert readiness["inputs"]["stage_a_dir"].endswith("stage_a_code_domain_v2_balanced")
    assert readiness["status"] == "candidate_pool_ready_for_v2_development_design"
    assert not readiness["blockers"]
    assert readiness["utility_scope"] == "Stage C validation only; never selector objective"
    print("[code-domain-v2-balanced-pool] balanced candidate pool readiness: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
