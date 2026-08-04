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
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class QualityActionDecision:
    action: QualityAction
    reason_code: str
    failed_policy_ids: tuple[str, ...]


def _is_unanimous_first_pass_fail(result: PanelPolicyResult) -> bool:
    return (
        result.decision is PanelDecision.FAIL
        and len(result.first_pass) == 3
        and all(vote.decision is PolicyDecision.FAIL for vote in result.first_pass)
    )


def decide_quality_action(
    results: tuple[PanelPolicyResult, ...],
    mode: CurationMode,
    coverage_veto: bool,
) -> QualityActionDecision:
    if mode is CurationMode.NORMAL:
        failed = tuple(
            result.policy_id for result in results if _is_unanimous_first_pass_fail(result)
        )
    else:
        failed = tuple(
            result.policy_id for result in results if result.decision is PanelDecision.FAIL
        )
    if not failed:
        return QualityActionDecision(QualityAction.RETAIN, "quality_no_qualified_fail", ())
    if coverage_veto:
        return QualityActionDecision(QualityAction.RETAIN, "coverage_veto_retain", failed)
    strength = "unanimous" if mode is CurationMode.NORMAL else "stable_majority"
    return QualityActionDecision(
        QualityAction.REMOVE,
        f"quality_{mode.value}_{strength}_fail",
        failed,
    )
