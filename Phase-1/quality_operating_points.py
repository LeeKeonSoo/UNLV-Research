from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from quality_teacher_panel import PanelDecision, PolicyDecision
from quality_teacher_runtime import PanelPolicyResult


class CurationMode(str, Enum):
    NORMAL = "normal"
    HARD = "hard"


class QualityAction(str, Enum):
    RETAIN = "retain"
    NOT_SELECT = "not_select"


@dataclass(frozen=True, slots=True)
class QualityActionDecision:
    action: QualityAction
    reason_code: str
    failed_policy_ids: tuple[str, ...]
    passed_policy_ids: tuple[str, ...]
    required_pass_count: int


def required_quality_pass_count(mode: CurationMode) -> int:
    return 1 if mode is CurationMode.NORMAL else 2


def _is_unanimous_first_pass_fail(result: PanelPolicyResult) -> bool:
    if result.decision_source == "distilled_ranker":
        return _is_distilled_fail(result, CurationMode.NORMAL)
    if result.decision_source == "declared_verifier":
        return result.decision is PanelDecision.FAIL
    return (
        result.decision is PanelDecision.FAIL
        and len(result.first_pass) == 3
        and all(vote.decision is PolicyDecision.FAIL for vote in result.first_pass)
    )


def _is_distilled_fail(result: PanelPolicyResult, mode: CurationMode) -> bool:
    if result.decision_source != "distilled_ranker" or result.out_of_distribution:
        return False
    threshold = (
        result.normal_failure_threshold
        if mode is CurationMode.NORMAL
        else result.hard_failure_threshold
    )
    return (
        threshold is not None
        and result.failure_probability is not None
        and result.failure_probability >= threshold
        and result.decision is PanelDecision.FAIL
    )


def decide_quality_action(
    results: tuple[PanelPolicyResult, ...],
    mode: CurationMode,
    coverage_veto: bool,
) -> QualityActionDecision:
    required_passes = required_quality_pass_count(mode)
    passed = tuple(
        result.policy_id
        for result in results
        if result.decision is PanelDecision.PASS and not result.out_of_distribution
    )
    if mode is CurationMode.NORMAL:
        failed = tuple(
            result.policy_id for result in results if _is_unanimous_first_pass_fail(result)
        )
    else:
        failed = tuple(
            result.policy_id
            for result in results
            if (
                _is_distilled_fail(result, mode)
                if result.decision_source == "distilled_ranker"
                else result.decision is PanelDecision.FAIL
            )
        )
    if coverage_veto:
        return QualityActionDecision(
            QualityAction.RETAIN,
            "coverage_veto_retain",
            failed,
            passed,
            required_passes,
        )
    if failed:
        return QualityActionDecision(
            QualityAction.NOT_SELECT,
            f"quality_{mode.value}_qualified_fail",
            failed,
            passed,
            required_passes,
        )
    if len(passed) < required_passes:
        return QualityActionDecision(
            QualityAction.NOT_SELECT,
            f"quality_{mode.value}_retention_threshold_not_met",
            (),
            passed,
            required_passes,
        )
    return QualityActionDecision(
        QualityAction.RETAIN,
        f"quality_{mode.value}_retention_threshold_met",
        (),
        passed,
        required_passes,
    )
