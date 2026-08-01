#!/usr/bin/env python3
"""Regression tests for Stage-C Utility baseline role separation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from policy.subsets import (
    CANONICAL_UTILITY_BASELINE,
    OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
    _diagnostic_matched_baseline_pools,
    _multi_matched_bucket_from_scored_record,
    _nuisance_matched_bucket_from_scored_record,
)


def _record(uid: str, quality: float, repeat: float) -> dict:
    return {
        "chunk_uid": uid,
        "source": "source-a",
        "provenance": {
            "input_source": "source-a",
            "metadata": {"domain": "science"},
        },
        "core_metrics": {
            "reference_quality_score": {"score": quality},
            "structural_validity_gate": {
                "details": {
                    "word_count": 128,
                    "style_bucket": "general_prose",
                }
            },
            "shingle_near_duplicate_risk_score": {
                "score": 0.2,
                "details": {"intra_chunk_repeat_pressure": repeat},
            },
        },
    }


def main() -> int:
    selected = _record("selected", 0.98, 0.31)
    lower_quality_same_nuisance = _record("control-same", 0.70, 0.31)
    different_repeat = _record("control-repeat", 0.70, 0.60)

    assert CANONICAL_UTILITY_BASELINE == "baseline_multi_matched_stageA_random"
    assert OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE == "baseline_nuisance_matched_stageA_random"
    assert (
        _nuisance_matched_bucket_from_scored_record(selected)
        == _nuisance_matched_bucket_from_scored_record(lower_quality_same_nuisance)
    )
    assert (
        _multi_matched_bucket_from_scored_record(selected)
        != _multi_matched_bucket_from_scored_record(lower_quality_same_nuisance)
    )
    assert (
        _nuisance_matched_bucket_from_scored_record(selected)
        != _nuisance_matched_bucket_from_scored_record(different_repeat)
    )

    pools = _diagnostic_matched_baseline_pools(
        baseline_records=[selected, lower_quality_same_nuisance, different_repeat],
        selected_records=[selected],
        seed=42,
        pool_multiplier=1,
        exclude_selected=True,
    )
    operational_candidate = pools[OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE]
    pool = operational_candidate["allowed_uids"]
    diagnostics = operational_candidate["diagnostics"]
    assert "selected" not in pool, pool
    assert "control-same" in pool, pool
    assert diagnostics["matched_variables"] == ["length", "style", "domain", "repeat_pressure"]
    assert diagnostics["excluded_selector_target_variables"] == ["quality", "redundancy_risk"]
    assert diagnostics["matching_policy"] == "exact_length_style_domain_repeat_pressure"
    assert diagnostics["fallback_order"] == []
    assert diagnostics["excluded_selected_records"] == 1
    print("[utility-baseline-contract] nuisance controls and selector-target exclusion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
