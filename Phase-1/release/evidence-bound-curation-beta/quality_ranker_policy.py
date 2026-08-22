from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quality_model_evidence import QualityDecision, QualityPolicyEvidence


@dataclass(frozen=True, slots=True)
class DistilledPolicyContract:
    policy_id: str
    class_labels: tuple[str, ...]
    failure_threshold: float | None
    minimum_decision_confidence: float
    ranker_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_id or not self.class_labels:
            raise ValueError("Distilled Quality policy identity is required")
        if set(self.class_labels) - {"pass", "fail", "abstain"}:
            raise ValueError("Distilled Quality policy classes are invalid")
        if not 0.0 < self.minimum_decision_confidence <= 1.0:
            raise ValueError("Decision confidence must be in (0, 1]")
        if self.failure_threshold is not None and not 0.0 <= self.failure_threshold <= 1.0:
            raise ValueError("Failure thresholds must be probabilities")
        if len(self.ranker_artifact_sha256) != 64:
            raise ValueError("Distilled Quality ranker identity must be frozen")


def distilled_policy_result(
    contract: DistilledPolicyContract,
    *,
    class_probabilities: NDArray[np.float64],
    out_of_distribution: bool,
) -> QualityPolicyEvidence:
    if class_probabilities.shape != (len(contract.class_labels),):
        raise ValueError("Distilled class probabilities do not match the policy contract")
    if np.any(class_probabilities < 0.0) or not np.isclose(class_probabilities.sum(), 1.0):
        raise ValueError("Distilled class probabilities must form a distribution")
    confidence = float(class_probabilities.max())
    predicted = contract.class_labels[int(class_probabilities.argmax())]
    if out_of_distribution:
        decision = QualityDecision.ABSTAIN
        reasons = ("quality_ranker_ood_abstain",)
    elif confidence < contract.minimum_decision_confidence:
        decision = QualityDecision.ABSTAIN
        reasons = ("quality_ranker_low_confidence_abstain",)
    else:
        decision = QualityDecision(predicted)
        reasons = (f"quality_ranker_{predicted}",)
    fail_probability = (
        float(class_probabilities[contract.class_labels.index("fail")])
        if "fail" in contract.class_labels
        else 0.0
    )
    return QualityPolicyEvidence(
        policy_id=contract.policy_id,
        decision=decision,
        reason_codes=reasons,
        class_probabilities=tuple(
            (label, float(probability))
            for label, probability in zip(
                contract.class_labels,
                class_probabilities,
                strict=True,
            )
        ),
        failure_probability=fail_probability,
        failure_threshold=contract.failure_threshold,
        prediction_confidence=confidence,
        minimum_decision_confidence=contract.minimum_decision_confidence,
        out_of_distribution=out_of_distribution,
        ranker_artifact_sha256=contract.ranker_artifact_sha256,
    )
