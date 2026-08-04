from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from quality_teacher_panel import (
    PanelDecision,
    PolicyDecision,
    QualityPolicy,
    TeacherPanel,
    TeacherSpec,
    TeacherVote,
    decide_panel,
)
from quality_teacher_response import (
    TeacherResponseAttempt,
    parse_teacher_response,
    resolve_teacher_response,
)


@dataclass(frozen=True, slots=True)
class TeacherGenerationUnavailable(RuntimeError):
    teacher_id: str
    reason: str

    def __str__(self) -> str:
        return f"Teacher generation unavailable for {self.teacher_id}: {self.reason}"


class EvaluationUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    declared_context: str | None = None
    attached_evidence: tuple[str, ...]


class TeacherGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    policy_name: str = Field(min_length=1)
    policy_question: str = Field(min_length=1)
    fail_boundary: str = Field(min_length=1)
    abstain_boundary: str = Field(min_length=1)
    pass_reason_codes: tuple[str, ...] = Field(min_length=1)
    fail_reason_codes: tuple[str, ...] = Field(min_length=1)
    abstain_reason_codes: tuple[str, ...] = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    unit_text: str = Field(min_length=1)
    declared_context: str | None
    attached_evidence: tuple[str, ...]
    pass_index: Literal[1, 2]
    blind_run_id: str = Field(min_length=1)
    schema_retry: bool


class TeacherAdapter(Protocol):
    def generate(self, request: TeacherGenerationRequest) -> str: ...


@dataclass(frozen=True, slots=True)
class PanelPolicyResult:
    policy_id: str
    decision: PanelDecision
    first_pass: tuple[TeacherVote, ...]
    second_pass: tuple[TeacherVote, ...] | None


def _request(
    teacher: TeacherSpec,
    policy: QualityPolicy,
    unit: EvaluationUnit,
    *,
    pass_index: Literal[1, 2],
    blind_run_id: str,
    schema_retry: bool,
) -> TeacherGenerationRequest:
    return TeacherGenerationRequest(
        teacher_id=teacher.teacher_id,
        model_id=teacher.model_id,
        policy_id=policy.policy_id,
        policy_name=policy.name,
        policy_question=policy.question,
        fail_boundary=policy.fail_boundary,
        abstain_boundary=policy.abstain_boundary,
        pass_reason_codes=policy.reason_codes.pass_,
        fail_reason_codes=policy.reason_codes.fail,
        abstain_reason_codes=policy.reason_codes.abstain,
        unit_id=unit.unit_id,
        unit_text=unit.text,
        declared_context=unit.declared_context,
        attached_evidence=unit.attached_evidence,
        pass_index=pass_index,
        blind_run_id=blind_run_id,
        schema_retry=schema_retry,
    )


def evaluate_teacher(
    adapter: TeacherAdapter,
    teacher: TeacherSpec,
    policy: QualityPolicy,
    unit: EvaluationUnit,
    *,
    pass_index: Literal[1, 2],
    blind_run_id: str | None = None,
) -> TeacherVote:
    run_id = blind_run_id or uuid4().hex
    first_request = _request(
        teacher,
        policy,
        unit,
        pass_index=pass_index,
        blind_run_id=run_id,
        schema_retry=False,
    )
    try:
        first_raw = adapter.generate(first_request)
    except TeacherGenerationUnavailable:
        return TeacherVote(
            teacher_id=teacher.teacher_id,
            policy_id=policy.policy_id,
            decision=PolicyDecision.ABSTAIN,
            reason_codes=("teacher_generation_unavailable",),
        )
    retry_raw: str | None = None
    first_payload = parse_teacher_response(first_raw)
    first_valid = (
        first_payload is not None
        and set(first_payload.reason_codes)
        <= set(policy.reason_codes.for_decision(first_payload.decision))
    )
    if not first_valid:
        try:
            retry_raw = adapter.generate(first_request.model_copy(update={"schema_retry": True}))
        except TeacherGenerationUnavailable:
            return TeacherVote(
                teacher_id=teacher.teacher_id,
                policy_id=policy.policy_id,
                decision=PolicyDecision.ABSTAIN,
                reason_codes=("teacher_generation_unavailable",),
            )
    return resolve_teacher_response(
        TeacherResponseAttempt(
            teacher_id=teacher.teacher_id,
            policy=policy,
            first_raw=first_raw,
            retry_raw=retry_raw,
        )
    )


def _run_pass(
    panel: TeacherPanel,
    adapters: Mapping[str, TeacherAdapter],
    policy: QualityPolicy,
    unit: EvaluationUnit,
    *,
    pass_index: Literal[1, 2],
    blind_run_id: str,
) -> tuple[TeacherVote, ...]:
    return tuple(
        evaluate_teacher(
            adapters[teacher.teacher_id],
            teacher,
            policy,
            unit,
            pass_index=pass_index,
            blind_run_id=blind_run_id,
        )
        for teacher in panel.teachers
    )


def evaluate_panel_policy(
    panel: TeacherPanel,
    adapters: Mapping[str, TeacherAdapter],
    policy: QualityPolicy,
    unit: EvaluationUnit,
) -> PanelPolicyResult:
    first_pass = _run_pass(
        panel,
        adapters,
        policy,
        unit,
        pass_index=1,
        blind_run_id=uuid4().hex,
    )
    first_decision = decide_panel(first_pass, None)
    if first_decision is not PanelDecision.ABSTAIN:
        return PanelPolicyResult(policy.policy_id, first_decision, first_pass, None)
    counts = Counter(vote.decision for vote in first_pass)
    non_abstain_majority = max(
        counts[PolicyDecision.PASS],
        counts[PolicyDecision.FAIL],
    )
    if non_abstain_majority != 2:
        return PanelPolicyResult(policy.policy_id, PanelDecision.ABSTAIN, first_pass, None)
    second_pass = _run_pass(
        panel,
        adapters,
        policy,
        unit,
        pass_index=2,
        blind_run_id=uuid4().hex,
    )
    return PanelPolicyResult(
        policy.policy_id,
        decide_panel(first_pass, second_pass),
        first_pass,
        second_pass,
    )


def evaluate_quality_unit(
    panel: TeacherPanel,
    adapters: Mapping[str, TeacherAdapter],
    unit: EvaluationUnit,
) -> tuple[PanelPolicyResult, ...]:
    return tuple(
        evaluate_panel_policy(panel, adapters, policy, unit)
        for policy in panel.policies
    )


def resolve_quality_gate(
    results: tuple[PanelPolicyResult, ...],
) -> PanelDecision:
    if any(result.decision is PanelDecision.FAIL for result in results):
        return PanelDecision.FAIL
    if results and all(result.decision is PanelDecision.PASS for result in results):
        return PanelDecision.PASS
    return PanelDecision.ABSTAIN
