#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_operating_points import QualityAction, decide_quality_action
from quality_model_evidence import (
    MissingQualityFallbackEvidenceError,
    QualityDecision,
    QualityPolicyEvidence,
)


def _result(
    policy_id: str,
    decision: QualityDecision,
    failure_probability: float = 0.0,
    prediction_confidence: float = 1.0,
) -> QualityPolicyEvidence:
    return QualityPolicyEvidence(
        policy_id=policy_id,
        decision=decision,
        reason_codes=(f"quality_ranker_{decision.value}",),
        class_probabilities=((decision.value, 1.0),),
        failure_probability=failure_probability,
        failure_threshold=0.7,
        prediction_confidence=prediction_confidence,
        minimum_decision_confidence=0.7,
        out_of_distribution=False,
        ranker_artifact_sha256="a" * 64,
    )


def test_substantive_positive_evidence_alone_requires_fallback() -> None:
    passing = _result("q3_substantive_payload", QualityDecision.PASS)

    try:
        decide_quality_action((passing,), False)
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Q3 alone must not authorize retention")


def test_coherence_payload_and_relations_jointly_retain() -> None:
    passing = tuple(
        _result(policy_id, QualityDecision.PASS)
        for policy_id in (
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        )
    )

    assert decide_quality_action(passing, False).action is QualityAction.RETAIN


def test_absence_of_positive_evidence_requires_fallback() -> None:
    abstained = _result("q4_learnable_relations", QualityDecision.ABSTAIN)

    try:
        decide_quality_action((abstained,), False)
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Abstention must not silently retain a chunk")


def test_only_qualified_fail_is_not_selected_and_coverage_may_restore() -> None:
    failed = _result(
        "q3_substantive_payload",
        QualityDecision.FAIL,
        0.9,
    )
    rejected = decide_quality_action((failed,), False)
    assert rejected.action is QualityAction.NOT_SELECT
    assert rejected.reason_code == "quality_qualified_fail"

    vetoed = decide_quality_action((failed,), True)
    assert vetoed.action is QualityAction.RETAIN
    assert vetoed.reason_code == "coverage_veto_retain"


def test_fail_below_frozen_threshold_retains() -> None:
    unqualified_fail = _result("q1_correctness_evidence", QualityDecision.FAIL, 0.65)

    try:
        decide_quality_action((unqualified_fail,), False)
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Unqualified failure must be resolved by fallback evidence")


def test_fail_below_frozen_confidence_retains() -> None:
    low_confidence_fail = _result(
        "q1_correctness_evidence", QualityDecision.FAIL, 0.9, 0.6
    )

    try:
        decide_quality_action((low_confidence_fail,), False)
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("Low-confidence failure must be resolved by fallback evidence")


if __name__ == "__main__":
    test_substantive_positive_evidence_alone_requires_fallback()
    test_coherence_payload_and_relations_jointly_retain()
    test_absence_of_positive_evidence_requires_fallback()
    test_only_qualified_fail_is_not_selected_and_coverage_may_restore()
    test_fail_below_frozen_threshold_retains()
    test_fail_below_frozen_confidence_retains()
    print("[quality-operating-points-v2] positive support and Luna fallback: pass")
