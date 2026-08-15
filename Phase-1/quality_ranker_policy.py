from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quality_teacher_panel import PanelDecision
from quality_teacher_runtime import PanelPolicyResult


@dataclass(frozen=True, slots=True)
class DistilledPolicyContract:
    policy_id: str
    class_labels: tuple[str, ...]
    normal_fail_threshold: float | None
    hard_fail_threshold: float | None
    minimum_decision_confidence: float
    ranker_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.policy_id or not self.class_labels:
            raise ValueError("Distilled Quality policy identity is required")
        if set(self.class_labels) - {"pass", "fail", "abstain"}:
            raise ValueError("Distilled Quality policy classes are invalid")
        if not 0.0 < self.minimum_decision_confidence <= 1.0:
            raise ValueError("Decision confidence must be in (0, 1]")
        thresholds = tuple(
            value
            for value in (self.normal_fail_threshold, self.hard_fail_threshold)
            if value is not None
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("Failure thresholds must be probabilities")
        if (
            self.normal_fail_threshold is not None
            and self.hard_fail_threshold is not None
            and self.normal_fail_threshold < self.hard_fail_threshold
        ):
            raise ValueError("Normal failure threshold cannot be weaker than Hard")
        if len(self.ranker_artifact_sha256) != 64:
            raise ValueError("Distilled Quality ranker identity must be frozen")


def calibrate_failure_threshold(
    labels: NDArray[np.int64],
    fail_probabilities: NDArray[np.float64],
    *,
    maximum_false_positive_rate: float,
    minimum_fail_predictions: int,
) -> float | None:
    """Choose the most inclusive observed threshold within an empirical FPR bound."""
    if labels.shape != fail_probabilities.shape or labels.ndim != 1:
        raise ValueError("Calibration labels and probabilities must be aligned vectors")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Failure calibration labels must be binary")
    if not 0.0 <= maximum_false_positive_rate < 1.0 or minimum_fail_predictions < 1:
        raise ValueError("Failure calibration constraints are invalid")
    negatives = labels == 0
    if not np.any(negatives):
        return None
    valid: list[float] = []
    for threshold in sorted(set(float(value) for value in fail_probabilities)):
        predicted = fail_probabilities >= threshold
        if int(np.count_nonzero(predicted)) < minimum_fail_predictions:
            continue
        false_positive_rate = float(np.count_nonzero(predicted & negatives)) / float(
            np.count_nonzero(negatives)
        )
        if false_positive_rate <= maximum_false_positive_rate:
            valid.append(threshold)
    return min(valid) if valid else None


def distilled_policy_result(
    contract: DistilledPolicyContract,
    *,
    class_probabilities: NDArray[np.float64],
    out_of_distribution: bool,
) -> PanelPolicyResult:
    if class_probabilities.shape != (len(contract.class_labels),):
        raise ValueError("Distilled class probabilities do not match the policy contract")
    if np.any(class_probabilities < 0.0) or not np.isclose(class_probabilities.sum(), 1.0):
        raise ValueError("Distilled class probabilities must form a distribution")
    confidence = float(class_probabilities.max())
    predicted = contract.class_labels[int(class_probabilities.argmax())]
    if out_of_distribution:
        decision = PanelDecision.ABSTAIN
        reasons = ("quality_ranker_ood_abstain",)
    elif confidence < contract.minimum_decision_confidence:
        decision = PanelDecision.ABSTAIN
        reasons = ("quality_ranker_low_confidence_abstain",)
    else:
        decision = PanelDecision(predicted)
        reasons = (f"quality_ranker_{predicted}",)
    fail_probability = (
        float(class_probabilities[contract.class_labels.index("fail")])
        if "fail" in contract.class_labels
        else 0.0
    )
    return PanelPolicyResult(
        policy_id=contract.policy_id,
        decision=decision,
        first_pass=(),
        second_pass=None,
        decision_source="distilled_ranker",
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
        normal_failure_threshold=contract.normal_fail_threshold,
        hard_failure_threshold=contract.hard_fail_threshold,
        prediction_confidence=confidence,
        out_of_distribution=out_of_distribution,
        ranker_artifact_sha256=contract.ranker_artifact_sha256,
    )
