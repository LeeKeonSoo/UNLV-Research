#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_consistency_audit import build_consistency_audit


def test_current_framework_is_internally_aligned_but_release_blocked() -> None:
    report = build_consistency_audit(ROOT)

    assert report["implementation_consistency"] == "passed"
    assert report["confirmatory_candidate_ready"] is False
    assert report["paper_claim_ready"] is False
    assert report["production_release_ready"] is False
    assert report["checks"]["semantic_coverage_typed_lineage"] is True
    assert report["checks"]["semantic_provider_lifecycle_alignment"] is True
    assert report["stage_ownership"] == {
        "stage_a": ["validity"],
        "stage_b": ["redundancy", "quality"],
        "stage_c": ["coverage"],
    }
    assert report["quality_ranker_policy_lifecycle"] == "candidate"
    assert "framework_release_blocked" in report["readiness_blockers"]
    assert "semantic_coverage_scientific_promotion_missing" in report["readiness_blockers"]


if __name__ == "__main__":
    test_current_framework_is_internally_aligned_but_release_blocked()
    print("[framework-consistency-audit-v2] aligned and fail-closed: pass")
