from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from quality_operating_points import (
    CurationMode,
    QualityAction,
    decide_quality_action,
)
from quality_teacher_runtime import PanelPolicyResult


@dataclass(frozen=True, slots=True)
class StagedQualityDecision:
    chunk_uid: str
    mode: str
    failed_policy_ids: tuple[str, ...]
    passed_policy_ids: tuple[str, ...]
    required_pass_count: int
    stage_b_action: str
    stage_b_reason_code: str
    final_action: str
    stage_c_reason_code: str
    benchmark_outcomes_read: bool = False
    utility_read: bool = False
    token_budget_read: bool = False


def propose_quality_selections(
    results_by_chunk: Mapping[str, tuple[PanelPolicyResult, ...]],
    mode: CurationMode,
) -> dict[str, StagedQualityDecision]:
    proposals: dict[str, StagedQualityDecision] = {}
    for chunk_uid, results in results_by_chunk.items():
        decision = decide_quality_action(results, mode, coverage_veto=False)
        proposals[chunk_uid] = StagedQualityDecision(
            chunk_uid=chunk_uid,
            mode=mode.value,
            failed_policy_ids=decision.failed_policy_ids,
            passed_policy_ids=decision.passed_policy_ids,
            required_pass_count=decision.required_pass_count,
            stage_b_action=decision.action.value,
            stage_b_reason_code=decision.reason_code,
            final_action=decision.action.value,
            stage_c_reason_code="coverage_not_required",
        )
    return proposals


def apply_coverage_veto(
    proposals: Mapping[str, StagedQualityDecision],
    protected_uids: set[str] | frozenset[str],
) -> dict[str, StagedQualityDecision]:
    return {
        chunk_uid: (
            replace(
                proposal,
                final_action=QualityAction.RETAIN.value,
                stage_c_reason_code="coverage_veto_retain",
            )
            if proposal.stage_b_action == QualityAction.NOT_SELECT.value
            and chunk_uid in protected_uids
            else replace(proposal, stage_c_reason_code="coverage_constraints_satisfied")
        )
        for chunk_uid, proposal in proposals.items()
    }
