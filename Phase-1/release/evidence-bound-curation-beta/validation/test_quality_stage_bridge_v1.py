#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_model_evidence import QualityDecision, QualityPolicyEvidence
from quality_stage_bridge import propose_quality_selections


def _failed_result() -> QualityPolicyEvidence:
    return QualityPolicyEvidence(
        policy_id="q3_substantive_payload",
        decision=QualityDecision.FAIL,
        reason_codes=("quality_ranker_fail",),
        class_probabilities=(("fail", 1.0),),
        failure_probability=1.0,
        failure_threshold=0.7,
        prediction_confidence=1.0,
        minimum_decision_confidence=0.7,
        out_of_distribution=False,
        ranker_artifact_sha256="a" * 64,
    )


def test_stage_b_emits_only_qualified_fail_nonselection() -> None:
    proposals = propose_quality_selections(
        {"keep-me": (_failed_result(),), "remove-me": (_failed_result(),)},
    )

    assert proposals["remove-me"].final_action == "not_select"
    assert proposals["keep-me"].final_action == "not_select"
    assert proposals["keep-me"].stage_b_reason_code == "quality_qualified_fail"
    assert proposals["keep-me"].stage_c_reason_code == "coverage_not_required"


if __name__ == "__main__":
    test_stage_b_emits_only_qualified_fail_nonselection()
    print("[quality-stage-bridge-v2] typed Quality membership authority: pass")
