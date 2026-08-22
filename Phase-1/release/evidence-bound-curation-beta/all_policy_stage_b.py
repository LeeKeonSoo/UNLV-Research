from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from quality_operating_points import QualityAction
from quality_model_evidence import (
    QualityPolicyEvidence,
    TeacherQualityPolicyEvidence,
    quality_evidence_to_mapping,
    teacher_quality_evidence_to_mapping,
)
from quality_stage_bridge import propose_quality_selections
from redundancy_equivalence import RedundancyMode
from redundancy_mode_policy import RedundancyPlan, build_redundancy_plan
from redundancy_v2 import RedundancySettings, RedundancyUnit


JsonMap = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RedundancyPolicyResult:
    survivors: tuple[JsonMap, ...]
    removals: tuple[JsonMap, ...]
    plan: RedundancyPlan
    audit: JsonMap


@dataclass(frozen=True, slots=True)
class QualityPolicyResult:
    survivors: tuple[JsonMap, ...]
    not_selected: tuple[JsonMap, ...]
    audit: JsonMap


def _removal_trace(row: Mapping[str, Any], reason_code: str) -> JsonMap:
    previous = row.get("stage_b_policy")
    return {
        "action": "remove",
        "removed_reason": reason_code,
        "prior_structural_trace": dict(previous) if isinstance(previous, dict) else None,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "token_budget_read": False,
    }


def apply_redundancy_policy(
    rows: tuple[JsonMap, ...] | list[JsonMap],
    *,
    mode: RedundancyMode,
    settings: RedundancySettings,
) -> RedundancyPolicyResult:
    units = tuple(
        RedundancyUnit(uid=str(row["chunk_uid"]), text=str(row["text"])) for row in rows
    )
    plan = build_redundancy_plan(units, settings, mode)
    proposal_by_uid = {proposal.removed_uid: proposal for proposal in plan.removals}
    survivors: list[JsonMap] = []
    removals: list[JsonMap] = []
    for source in rows:
        row = dict(source)
        uid = str(row["chunk_uid"])
        proposal = proposal_by_uid.get(uid)
        if proposal is None:
            row["stage_b_redundancy_v2"] = {
                "action": "retain",
                "mode": mode.value,
                "reason_code": "redundancy_no_authorized_equivalent",
                "benchmark_outcomes_read": False,
                "utility_read": False,
            }
            survivors.append(row)
            continue
        trace = {
            "action": "remove",
            "mode": mode.value,
            "reason_code": proposal.reason_code,
            "representative_chunk_uid": proposal.representative_uid,
            "family_id": proposal.family_id,
            "witness_kind": proposal.witness_kind.value,
            "evidence_sha256": proposal.evidence_sha256,
            "removed_token_count": proposal.removed_token_count,
            "coverage_veto_required": proposal.coverage_veto_required,
            "benchmark_outcomes_read": proposal.benchmark_outcomes_read,
            "utility_read": proposal.utility_read,
        }
        row["stage_b_redundancy_v2"] = trace
        row["stage_b_policy"] = {
            **_removal_trace(row, proposal.reason_code),
            "representative_chunk_uid": proposal.representative_uid,
            "family_id": proposal.family_id,
            "witness_kind": proposal.witness_kind.value,
            "evidence_sha256": proposal.evidence_sha256,
        }
        removals.append(row)
    witness_counts = Counter(proposal.witness_kind.value for proposal in plan.removals)
    retrieval_reason_counts = Counter(
        reason
        for pair in plan.candidate_pairs
        for reason in pair.retrieval_reasons
    )
    relation_counts = Counter(
        decision.relation.relation.value for decision in plan.authority_decisions
    )
    return RedundancyPolicyResult(
        survivors=tuple(survivors),
        removals=tuple(removals),
        plan=plan,
        audit={
            "schema_version": "stage-b-redundancy-v2-runtime-audit-v1",
            "mode": mode.value,
            "input_chunks": len(rows),
            "candidate_pairs_evaluated": len(plan.authority_decisions),
            "removed_chunks": len(removals),
            "retained_chunks": len(survivors),
            "family_count": len(plan.families),
            "removal_witness_counts": dict(sorted(witness_counts.items())),
            "relation_counts": dict(sorted(relation_counts.items())),
            "retrieval_reason_counts": dict(sorted(retrieval_reason_counts.items())),
            "candidate_only_pairs": sum(
                decision.witness is None for decision in plan.authority_decisions
            ),
            "all_removals_have_representative": all(
                proposal.representative_uid for proposal in plan.removals
            ),
            "benchmark_outcomes_read": False,
            "utility_read": False,
            "token_budget_read": False,
        },
    )


def apply_quality_policy(
    rows: tuple[JsonMap, ...] | list[JsonMap],
    *,
    results_by_chunk: Mapping[str, tuple[QualityPolicyEvidence, ...]],
    teacher_results_by_chunk: Mapping[
        str, tuple[TeacherQualityPolicyEvidence, ...]
    ] | None = None,
) -> QualityPolicyResult:
    input_uids = {str(row["chunk_uid"]) for row in rows}
    if set(results_by_chunk) != input_uids:
        raise RuntimeError("Quality results must cover every Stage-B input chunk exactly once")
    decisions = propose_quality_selections(
        results_by_chunk,
        teacher_results_by_chunk=teacher_results_by_chunk,
    )
    survivors: list[JsonMap] = []
    not_selected: list[JsonMap] = []
    reason_counts: Counter[str] = Counter()
    failed_policy_counts: Counter[str] = Counter()
    panel_decision_counts: Counter[str] = Counter()
    decision_source_counts: Counter[str] = Counter()
    chunks_with_any_abstain = 0
    chunks_with_all_policy_pass = 0
    for source in rows:
        row = dict(source)
        uid = str(row["chunk_uid"])
        decision = decisions[uid]
        row["quality_policy_evidence"] = [
            quality_evidence_to_mapping(result) for result in results_by_chunk[uid]
        ]
        teacher_results = (
            None
            if teacher_results_by_chunk is None
            else teacher_results_by_chunk.get(uid)
        )
        row["quality_teacher_evidence"] = (
            []
            if teacher_results is None
            else [teacher_quality_evidence_to_mapping(result) for result in teacher_results]
        )
        decision_source_counts[decision.decision_source] += 1
        panel_decision_counts.update(result.decision.value for result in results_by_chunk[uid])
        if any(result.decision.value == "abstain" for result in results_by_chunk[uid]):
            chunks_with_any_abstain += 1
        if all(result.decision.value == "pass" for result in results_by_chunk[uid]):
            chunks_with_all_policy_pass += 1
        quality_stage_decision = asdict(decision)
        quality_stage_decision["failed_policy_ids"] = list(decision.failed_policy_ids)
        quality_stage_decision["passed_policy_ids"] = list(decision.passed_policy_ids)
        quality_stage_decision["abstained_policy_ids"] = list(decision.abstained_policy_ids)
        row["quality_stage_decision"] = quality_stage_decision
        if decision.stage_b_action == QualityAction.NOT_SELECT.value:
            row["stage_b_policy"] = {
                **_removal_trace(row, decision.stage_b_reason_code),
                "action": "not_select",
                "failed_policy_ids": list(decision.failed_policy_ids),
                "passed_policy_ids": list(decision.passed_policy_ids),
                "abstained_policy_ids": list(decision.abstained_policy_ids),
            }
            not_selected.append(row)
            reason_counts[decision.stage_b_reason_code] += 1
            failed_policy_counts.update(decision.failed_policy_ids)
        else:
            survivors.append(row)
    return QualityPolicyResult(
        survivors=tuple(survivors),
        not_selected=tuple(not_selected),
        audit={
            "schema_version": "stage-b-quality-runtime-audit-v3",
            "decision_rule": "positive_support_with_luna_fallback",
            "input_chunks": len(rows),
            "retained_chunks": len(survivors),
            "not_selected_chunks": len(not_selected),
            "reason_code_counts": dict(sorted(reason_counts.items())),
            "failed_policy_counts": dict(sorted(failed_policy_counts.items())),
            "panel_policy_decision_counts": dict(sorted(panel_decision_counts.items())),
            "decision_source_counts": dict(sorted(decision_source_counts.items())),
            "teacher_reviewed_chunks": sum(
                1
                for uid in input_uids
                if teacher_results_by_chunk is not None and uid in teacher_results_by_chunk
            ),
            "chunks_with_any_abstain": chunks_with_any_abstain,
            "chunks_with_all_policy_pass": chunks_with_all_policy_pass,
            "chunks_without_qualified_fail": sum(
                not decision.failed_policy_ids for decision in decisions.values()
            ),
            "all_input_chunks_received_quality_decision": set(decisions) == input_uids,
            "abstain_action": "luna_fallback_then_not_select_without_positive_support",
            "benchmark_outcomes_read": False,
            "utility_read": False,
            "token_budget_read": False,
        },
    )


__all__ = [
    "QualityPolicyResult",
    "RedundancyPolicyResult",
    "apply_quality_policy",
    "apply_redundancy_policy",
]
