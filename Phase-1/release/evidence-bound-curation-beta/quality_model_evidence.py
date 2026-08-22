from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QualityEvidenceError(RuntimeError):
    """Raised when distilled Quality evidence violates the runtime contract."""


@dataclass(frozen=True, slots=True)
class MissingQualityFallbackEvidenceError(RuntimeError):
    policy_ids: tuple[str, ...]
    chunk_uid: str | None = None

    def __str__(self) -> str:
        scope = "" if self.chunk_uid is None else f" for {self.chunk_uid}"
        return f"Quality fallback evidence is required{scope}: {','.join(self.policy_ids)}"


class QualityDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class QualityPolicyEvidence:
    policy_id: str
    decision: QualityDecision
    reason_codes: tuple[str, ...]
    class_probabilities: tuple[tuple[str, float], ...]
    failure_probability: float | None
    failure_threshold: float | None
    prediction_confidence: float | None
    minimum_decision_confidence: float
    out_of_distribution: bool
    ranker_artifact_sha256: str
    decision_source: Literal["distilled_ranker"] = "distilled_ranker"

    def __post_init__(self) -> None:
        if not self.policy_id or not self.reason_codes:
            raise QualityEvidenceError("Quality evidence requires policy and reason identifiers")
        if not SHA256_RE.fullmatch(self.ranker_artifact_sha256):
            raise QualityEvidenceError("Quality evidence requires a frozen ranker artifact")
        if not 0.0 < self.minimum_decision_confidence <= 1.0:
            raise QualityEvidenceError("Quality evidence requires a frozen confidence threshold")


@dataclass(frozen=True, slots=True)
class TeacherQualityPolicyEvidence:
    policy_id: str
    decision: QualityDecision
    reason_codes: tuple[str, ...]
    observation_sha256: str
    decision_source: Literal["luna_fallback"] = "luna_fallback"

    def __post_init__(self) -> None:
        if not self.policy_id or not self.reason_codes:
            raise QualityEvidenceError("Teacher Quality evidence requires policy and reason identifiers")
        if not SHA256_RE.fullmatch(self.observation_sha256):
            raise QualityEvidenceError("Teacher Quality evidence requires a frozen observation artifact")


def quality_evidence_to_mapping(evidence: QualityPolicyEvidence) -> dict[str, object]:
    return {
        "policy_id": evidence.policy_id,
        "panel_decision": evidence.decision.value,
        "decision_source": evidence.decision_source,
        "decision_reason_codes": list(evidence.reason_codes),
        "class_probabilities": dict(evidence.class_probabilities),
        "failure_probability": evidence.failure_probability,
        "failure_threshold": evidence.failure_threshold,
        "prediction_confidence": evidence.prediction_confidence,
        "minimum_decision_confidence": evidence.minimum_decision_confidence,
        "out_of_distribution": evidence.out_of_distribution,
        "ranker_artifact_sha256": evidence.ranker_artifact_sha256,
        "first_pass": [],
        "second_pass": None,
    }


def quality_evidence_from_mapping(payload: dict[str, object]) -> QualityPolicyEvidence:
    probabilities = payload.get("class_probabilities")
    if not isinstance(probabilities, dict):
        raise QualityEvidenceError("Quality evidence probabilities are missing")
    return QualityPolicyEvidence(
        policy_id=str(payload["policy_id"]),
        decision=QualityDecision(str(payload["panel_decision"])),
        reason_codes=tuple(str(code) for code in payload["decision_reason_codes"]),
        class_probabilities=tuple(
            (str(label), float(probability))
            for label, probability in probabilities.items()
        ),
        failure_probability=(
            None
            if payload.get("failure_probability") is None
            else float(payload["failure_probability"])
        ),
        failure_threshold=(
            None
            if payload.get("failure_threshold") is None
            else float(payload["failure_threshold"])
        ),
        prediction_confidence=(
            None
            if payload.get("prediction_confidence") is None
            else float(payload["prediction_confidence"])
        ),
        minimum_decision_confidence=float(payload["minimum_decision_confidence"]),
        out_of_distribution=bool(payload["out_of_distribution"]),
        ranker_artifact_sha256=str(payload["ranker_artifact_sha256"]),
    )


def teacher_quality_evidence_to_mapping(
    evidence: TeacherQualityPolicyEvidence,
) -> dict[str, object]:
    return {
        "policy_id": evidence.policy_id,
        "panel_decision": evidence.decision.value,
        "decision_source": evidence.decision_source,
        "decision_reason_codes": list(evidence.reason_codes),
        "observation_sha256": evidence.observation_sha256,
    }


__all__ = [
    "MissingQualityFallbackEvidenceError",
    "QualityDecision",
    "QualityEvidenceError",
    "QualityPolicyEvidence",
    "TeacherQualityPolicyEvidence",
    "quality_evidence_to_mapping",
    "quality_evidence_from_mapping",
    "teacher_quality_evidence_to_mapping",
]
