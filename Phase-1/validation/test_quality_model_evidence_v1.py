#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_model_evidence import (
    QualityDecision,
    QualityPolicyEvidence,
    quality_evidence_to_mapping,
)


def test_distilled_quality_evidence_has_no_runtime_teacher_dependency() -> None:
    evidence = QualityPolicyEvidence(
        policy_id="q3_substantive_payload",
        decision=QualityDecision.PASS,
        reason_codes=("quality_ranker_pass",),
        class_probabilities=(("pass", 0.9), ("fail", 0.1)),
        failure_probability=0.1,
        failure_threshold=0.7,
        prediction_confidence=0.9,
        minimum_decision_confidence=0.7,
        out_of_distribution=False,
        ranker_artifact_sha256="a" * 64,
    )

    serialized = quality_evidence_to_mapping(evidence)

    assert serialized["panel_decision"] == "pass"
    assert serialized["decision_source"] == "distilled_ranker"
    assert serialized["first_pass"] == []
    assert serialized["second_pass"] is None


if __name__ == "__main__":
    test_distilled_quality_evidence_has_no_runtime_teacher_dependency()
    print("[quality-model-evidence-v1] distilled runtime contract: pass")
