from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from quality_operating_points import decide_quality_action
from quality_model_evidence import (
    MissingQualityFallbackEvidenceError,
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
)


@dataclass(frozen=True, slots=True)
class StagedQualityDecision:
    chunk_uid: str
    failed_policy_ids: tuple[str, ...]
    passed_policy_ids: tuple[str, ...]
    abstained_policy_ids: tuple[str, ...]
    stage_b_action: str
    stage_b_reason_code: str
    final_action: str
    stage_c_reason_code: str
    decision_source: str
    benchmark_outcomes_read: bool = False
    utility_read: bool = False
    token_budget_read: bool = False


def propose_quality_selections(
    results_by_chunk: Mapping[str, tuple[QualityPolicyEvidence, ...]],
    teacher_results_by_chunk: Mapping[
        str, tuple[TeacherQualityPolicyEvidence, ...]
    ] | None = None,
) -> dict[str, StagedQualityDecision]:
    proposals: dict[str, StagedQualityDecision] = {}
    for chunk_uid, results in results_by_chunk.items():
        teacher_results = (
            None
            if teacher_results_by_chunk is None
            else teacher_results_by_chunk.get(chunk_uid)
        )
        try:
            decision = decide_quality_action(
                results,
                coverage_veto=False,
                teacher_results=teacher_results,
            )
        except MissingQualityFallbackEvidenceError as error:
            raise MissingQualityFallbackEvidenceError(
                policy_ids=error.policy_ids,
                chunk_uid=chunk_uid,
            ) from error
        proposals[chunk_uid] = StagedQualityDecision(
            chunk_uid=chunk_uid,
            failed_policy_ids=decision.failed_policy_ids,
            passed_policy_ids=decision.passed_policy_ids,
            abstained_policy_ids=decision.abstained_policy_ids,
            stage_b_action=decision.action.value,
            stage_b_reason_code=decision.reason_code,
            final_action=decision.action.value,
            stage_c_reason_code="coverage_not_required",
            decision_source=decision.decision_source,
        )
    return proposals
