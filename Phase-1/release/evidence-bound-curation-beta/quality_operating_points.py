from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from quality_model_evidence import (
    MissingQualityFallbackEvidenceError,
    QualityDecision,
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
)


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


class QualityAction(StrEnum):
    RETAIN = "retain"
    NOT_SELECT = "not_select"


@dataclass(frozen=True, slots=True)
class QualityActionDecision:
    action: QualityAction
    reason_code: str
    failed_policy_ids: tuple[str, ...]
    passed_policy_ids: tuple[str, ...]
    abstained_policy_ids: tuple[str, ...]
    decision_source: str


def _is_qualified_fail(result: QualityPolicyEvidence) -> bool:
    if result.out_of_distribution:
        return False
    threshold = result.failure_threshold
    return (
        threshold is not None
        and result.failure_probability is not None
        and result.failure_probability >= threshold
        and result.prediction_confidence is not None
        and result.prediction_confidence >= result.minimum_decision_confidence
        and result.decision is QualityDecision.FAIL
    )


def _has_positive_support(passed_policy_ids: tuple[str, ...]) -> bool:
    passed = frozenset(passed_policy_ids)
    return {
        "q2_semantic_coherence",
        "q3_substantive_payload",
        "q4_learnable_relations",
    }.issubset(passed)


def quality_requires_teacher(results: tuple[QualityPolicyEvidence, ...]) -> bool:
    failed = any(_is_qualified_fail(result) for result in results)
    passed = tuple(
        result.policy_id
        for result in results
        if result.decision is QualityDecision.PASS and not result.out_of_distribution
    )
    return not failed and not _has_positive_support(passed)


def decide_quality_action(
    results: tuple[QualityPolicyEvidence, ...],
    coverage_veto: bool,
    teacher_results: tuple[TeacherQualityPolicyEvidence, ...] | None = None,
) -> QualityActionDecision:
    passed = tuple(
        result.policy_id
        for result in results
        if result.decision is QualityDecision.PASS and not result.out_of_distribution
    )
    failed = tuple(
        result.policy_id for result in results if _is_qualified_fail(result)
    )
    abstained = tuple(
        result.policy_id
        for result in results
        if result.decision is QualityDecision.ABSTAIN or result.out_of_distribution
    )
    if coverage_veto:
        return QualityActionDecision(
            QualityAction.RETAIN,
            "coverage_veto_retain",
            failed,
            passed,
            abstained,
            "coverage",
        )
    if failed:
        return QualityActionDecision(
            QualityAction.NOT_SELECT,
            "quality_qualified_fail",
            failed,
            passed,
            abstained,
            "distilled_ranker",
        )
    if _has_positive_support(passed):
        return QualityActionDecision(
            QualityAction.RETAIN,
            "quality_local_positive_support",
            (),
            passed,
            abstained,
            "distilled_ranker",
        )
    if teacher_results is None:
        raise MissingQualityFallbackEvidenceError(QUALITY_POLICY_IDS)
    teacher_ids = tuple(result.policy_id for result in teacher_results)
    if len(teacher_ids) != len(QUALITY_POLICY_IDS) or set(teacher_ids) != set(QUALITY_POLICY_IDS):
        raise MissingQualityFallbackEvidenceError(QUALITY_POLICY_IDS)
    teacher_failed = tuple(
        result.policy_id
        for result in teacher_results
        if result.decision is QualityDecision.FAIL
    )
    teacher_passed = tuple(
        result.policy_id
        for result in teacher_results
        if result.decision is QualityDecision.PASS
    )
    teacher_abstained = tuple(
        result.policy_id
        for result in teacher_results
        if result.decision is QualityDecision.ABSTAIN
    )
    if teacher_failed:
        return QualityActionDecision(
            QualityAction.NOT_SELECT,
            "quality_teacher_confirmed_fail",
            teacher_failed,
            teacher_passed,
            teacher_abstained,
            "luna_fallback",
        )
    if _has_positive_support(teacher_passed):
        return QualityActionDecision(
            QualityAction.RETAIN,
            "quality_teacher_positive_support",
            (),
            teacher_passed,
            teacher_abstained,
            "luna_fallback",
        )
    return QualityActionDecision(
        QualityAction.NOT_SELECT,
        "quality_teacher_no_positive_support",
        (),
        teacher_passed,
        teacher_abstained,
        "luna_fallback",
    )
