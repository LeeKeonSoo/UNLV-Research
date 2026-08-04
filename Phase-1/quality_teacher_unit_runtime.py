from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quality_teacher_adapters import (
    ChatMessage,
    CompletionBackend,
    CompletionRequest,
    StructuredResponseFormat,
)
from quality_teacher_panel import (
    PanelDecision,
    PolicyDecision,
    QualityPolicy,
    TeacherPanel,
    TeacherSpec,
    TeacherVote,
    decide_panel,
)
from quality_teacher_runtime import (
    EvaluationUnit,
    PanelPolicyResult,
    TeacherGenerationUnavailable,
)


@dataclass(frozen=True, slots=True)
class InsufficientTeacherAvailability(RuntimeError):
    unit_id: str
    available_teachers: int

    def __str__(self) -> str:
        return f"Only {self.available_teachers} teachers were available for {self.unit_id}"


class PolicySetGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    policies: tuple[QualityPolicy, QualityPolicy, QualityPolicy, QualityPolicy]
    unit: EvaluationUnit
    pass_index: Literal[1, 2]
    blind_run_id: str = Field(min_length=1)
    schema_retry: bool


class PolicySetAdapter(Protocol):
    def generate_policy_set(self, request: PolicySetGenerationRequest) -> str: ...


class PolicyVotePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    decision: PolicyDecision
    reason_codes: tuple[str, ...] = Field(min_length=1, max_length=8)


class PolicySetPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policies: tuple[PolicyVotePayload, ...] = Field(min_length=4, max_length=4)


@dataclass(frozen=True, slots=True)
class CombinedUnitResult:
    policy_results: tuple[PanelPolicyResult, ...]
    available_teacher_ids: tuple[str, ...]
    unavailable_teacher_ids: tuple[str, ...]


def parse_policy_set_response(
    raw: str,
    policies: tuple[QualityPolicy, ...],
) -> tuple[PolicyVotePayload, ...] | None:
    try:
        payload = PolicySetPayload.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None
    expected = {policy.policy_id: policy for policy in policies}
    observed = {vote.policy_id: vote for vote in payload.policies}
    if len(observed) != len(payload.policies) or set(observed) != set(expected):
        return None
    for policy_id, vote in observed.items():
        allowed = set(expected[policy_id].reason_codes.for_decision(vote.decision))
        if not set(vote.reason_codes) <= allowed:
            return None
    return tuple(observed[policy.policy_id] for policy in policies)


def _messages(request: PolicySetGenerationRequest) -> tuple[ChatMessage, ChatMessage]:
    policy_payloads = [
        {
            "policy_id": policy.policy_id,
            "name": policy.name,
            "question": policy.question,
            "fail_boundary": policy.fail_boundary,
            "abstain_boundary": policy.abstain_boundary,
            "allowed_reason_codes_by_decision": {
                "pass": list(policy.reason_codes.pass_),
                "fail": list(policy.reason_codes.fail),
                "abstain": list(policy.reason_codes.abstain),
            },
        }
        for policy in request.policies
    ]
    payload = {
        "policies": policy_payloads,
        "unit": request.unit.model_dump(mode="json"),
        "execution": {
            "pass_index": request.pass_index,
            "blind_run_id": request.blind_run_id,
            "schema_retry": request.schema_retry,
            "prior_panel_votes_available": False,
        },
        "response_contract": {
            "root_key": "policies",
            "one_result_per_supplied_policy": True,
            "result_keys": ["policy_id", "decision", "reason_codes"],
            "additional_properties": False,
        },
    }
    retry = "The previous response was invalid. " if request.schema_retry else ""
    system = (
        "You are one independent evaluator in a data-curation panel. Evaluate every supplied "
        "policy independently using only observable unit evidence. A pass on one policy cannot "
        "repair a fail on another. Use abstain whenever its boundary applies. "
        f"{retry}Return one schema-valid JSON object only."
    )
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=True, sort_keys=True)),
    )


@dataclass(frozen=True, slots=True)
class HostedPolicySetAdapter:
    teacher: TeacherSpec
    backend: CompletionBackend

    def generate_policy_set(self, request: PolicySetGenerationRequest) -> str:
        if request.teacher_id != self.teacher.teacher_id or request.model_id != self.teacher.model_id:
            raise TeacherGenerationUnavailable(request.teacher_id, "adapter_identity_mismatch")
        reason_codes = tuple(code for policy in request.policies for code in policy.reason_codes.all())
        return self.backend.complete(
            CompletionRequest(
                model_id=request.model_id,
                messages=_messages(request),
                maximum_new_tokens=min(256, self.teacher.maximum_new_tokens * 2),
                response_format=StructuredResponseFormat(
                    type="json_object",
                    allowed_reason_codes=reason_codes,
                ),
                temperature=self.teacher.temperature,
                top_p=self.teacher.top_p,
                reasoning_control=self.teacher.reasoning_control,
            )
        )


def _unavailable_votes(teacher: TeacherSpec, policies: tuple[QualityPolicy, ...]) -> tuple[TeacherVote, ...]:
    return tuple(
        TeacherVote(
            teacher_id=teacher.teacher_id,
            policy_id=policy.policy_id,
            decision=PolicyDecision.ABSTAIN,
            reason_codes=("teacher_generation_unavailable",),
        )
        for policy in policies
    )


def _evaluate_teacher_set(
    adapter: PolicySetAdapter,
    teacher: TeacherSpec,
    policies: tuple[QualityPolicy, ...],
    unit: EvaluationUnit,
    pass_index: Literal[1, 2],
) -> tuple[TeacherVote, ...]:
    request = PolicySetGenerationRequest(
        teacher_id=teacher.teacher_id,
        model_id=teacher.model_id,
        policies=policies,
        unit=unit,
        pass_index=pass_index,
        blind_run_id=uuid4().hex,
        schema_retry=False,
    )
    for schema_retry in (False, True):
        try:
            raw = adapter.generate_policy_set(request.model_copy(update={"schema_retry": schema_retry}))
        except TeacherGenerationUnavailable:
            return _unavailable_votes(teacher, policies)
        parsed = parse_policy_set_response(raw, policies)
        if parsed is not None:
            return tuple(
                TeacherVote(
                    teacher_id=teacher.teacher_id,
                    policy_id=vote.policy_id,
                    decision=vote.decision,
                    reason_codes=vote.reason_codes,
                )
                for vote in parsed
            )
    return tuple(
        TeacherVote(
            teacher_id=teacher.teacher_id,
            policy_id=policy.policy_id,
            decision=PolicyDecision.ABSTAIN,
            reason_codes=("invalid_teacher_response_schema",),
        )
        for policy in policies
    )


def _run_pass(
    panel: TeacherPanel,
    adapters: Mapping[str, PolicySetAdapter],
    unit: EvaluationUnit,
    pass_index: Literal[1, 2],
) -> tuple[tuple[TeacherVote, ...], ...]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        return tuple(
            executor.map(
                lambda teacher: _evaluate_teacher_set(
                    adapters[teacher.teacher_id], teacher, panel.policies, unit, pass_index
                ),
                panel.teachers,
            )
        )


def evaluate_quality_unit_combined(
    panel: TeacherPanel,
    adapters: Mapping[str, PolicySetAdapter],
    unit: EvaluationUnit,
) -> CombinedUnitResult:
    first_by_teacher = _run_pass(panel, adapters, unit, 1)
    available = tuple(
        teacher.teacher_id
        for teacher, votes in zip(panel.teachers, first_by_teacher, strict=True)
        if not all("teacher_generation_unavailable" in vote.reason_codes for vote in votes)
    )
    if len(available) < 2:
        raise InsufficientTeacherAvailability(unit.unit_id, len(available))
    first_by_policy = tuple(tuple(votes[index] for votes in first_by_teacher) for index in range(4))
    needs_second = tuple(
        decide_panel(votes, None) is PanelDecision.ABSTAIN
        and max(Counter(vote.decision for vote in votes)[decision] for decision in (PolicyDecision.PASS, PolicyDecision.FAIL)) == 2
        for votes in first_by_policy
    )
    second_by_policy: tuple[tuple[TeacherVote, ...], ...] | None = None
    if any(needs_second):
        second_by_teacher = _run_pass(panel, adapters, unit, 2)
        second_by_policy = tuple(tuple(votes[index] for votes in second_by_teacher) for index in range(4))
    results: list[PanelPolicyResult] = []
    for index, policy in enumerate(panel.policies):
        if policy.policy_id == "q1_correctness_evidence" and unit.declared_verifier is not None:
            passed = unit.declared_verifier.status == "pass"
            results.append(
                PanelPolicyResult(
                    policy.policy_id,
                    PanelDecision.PASS if passed else PanelDecision.FAIL,
                    (),
                    None,
                    "declared_verifier",
                    ("observable_correctness_evidence" if passed else "declared_verifier_failed",),
                )
            )
            continue
        second = second_by_policy[index] if needs_second[index] and second_by_policy else None
        results.append(
            PanelPolicyResult(
                policy.policy_id,
                decide_panel(first_by_policy[index], second),
                first_by_policy[index],
                second,
            )
        )
    unavailable = tuple(teacher.teacher_id for teacher in panel.teachers if teacher.teacher_id not in available)
    return CombinedUnitResult(tuple(results), available, unavailable)
