from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Final, Literal, Mapping, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from quality_teacher_adapters import ChatMessage, CompletionBackend, CompletionRequest, StructuredResponseFormat
from quality_teacher_json import parse_unique_json_model
from quality_teacher_panel import (
    PanelDecision,
    PolicyDecision,
    QualityPolicy,
    TeacherPanel,
    TeacherSpec,
    TeacherVote,
    decide_panel,
)
from quality_teacher_runtime import EvaluationUnit, PanelPolicyResult, TeacherGenerationUnavailable
from quality_teacher_unit_runtime import (
    CombinedUnitResult,
    InsufficientTeacherAvailability,
    PolicyVotePayload,
)


class PolicySetBatchGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    policies: tuple[QualityPolicy, QualityPolicy, QualityPolicy, QualityPolicy]
    units: tuple[EvaluationUnit, ...] = Field(min_length=1, max_length=16)
    pass_index: Literal[1, 2]
    blind_run_id: str = Field(min_length=1)
    schema_retry: bool


class PolicySetBatchAdapter(Protocol):
    def generate_policy_batch(self, request: PolicySetBatchGenerationRequest) -> str: ...


class TeacherBatchEvidenceStoreProtocol(Protocol):
    def get(
        self,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
    ) -> dict[str, tuple[TeacherVote, ...]] | None: ...

    def put(
        self,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
        votes_by_unit: Mapping[str, tuple[TeacherVote, ...]],
    ) -> None: ...


class UnitPolicyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    policies: tuple[PolicyVotePayload, ...] = Field(min_length=4, max_length=4)


class PolicySetBatchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    units: tuple[UnitPolicyPayload, ...] = Field(min_length=1, max_length=16)


@dataclass(frozen=True, slots=True)
class UnitBatchResult:
    unit_id: str
    evidence: CombinedUnitResult


NON_EVIDENCE_REASON_CODES: Final = frozenset(
    {"teacher_generation_unavailable", "invalid_teacher_response_schema"}
)


def parse_policy_set_batch_response(
    raw: str,
    units: tuple[EvaluationUnit, ...],
    policies: tuple[QualityPolicy, ...],
) -> dict[str, tuple[PolicyVotePayload, ...]] | None:
    payload = parse_unique_json_model(raw, PolicySetBatchPayload)
    if payload is None:
        return None
    expected_units = {unit.unit_id for unit in units}
    observed_units = {unit.unit_id: unit for unit in payload.units}
    if len(observed_units) != len(payload.units) or set(observed_units) != expected_units:
        return None
    expected_policies = {policy.policy_id: policy for policy in policies}
    result: dict[str, tuple[PolicyVotePayload, ...]] = {}
    for unit in units:
        votes = observed_units[unit.unit_id].policies
        by_policy = {vote.policy_id: vote for vote in votes}
        if len(by_policy) != len(votes) or set(by_policy) != set(expected_policies):
            return None
        for policy_id, vote in by_policy.items():
            allowed = set(expected_policies[policy_id].reason_codes.for_decision(vote.decision))
            if not set(vote.reason_codes) <= allowed:
                return None
        result[unit.unit_id] = tuple(by_policy[policy.policy_id] for policy in policies)
    return result


def _messages(request: PolicySetBatchGenerationRequest) -> tuple[ChatMessage, ChatMessage]:
    payload = {
        "policies": [
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
        ],
        "units": [unit.model_dump(mode="json") for unit in request.units],
        "execution": {
            "pass_index": request.pass_index,
            "blind_run_id": request.blind_run_id,
            "schema_retry": request.schema_retry,
            "prior_panel_votes_available": False,
        },
        "response_contract": {
            "root_key": "units",
            "one_result_per_supplied_unit": True,
            "unit_keys": ["unit_id", "policies"],
            "one_result_per_supplied_policy": True,
            "policy_result_keys": ["policy_id", "decision", "reason_codes"],
            "additional_properties": False,
        },
    }
    retry = "The previous response was invalid. " if request.schema_retry else ""
    system = (
        "You are one independent evaluator in a data-curation panel. Evaluate every unit and "
        "every supplied policy independently using only observable unit evidence. Decisions for "
        "one unit or policy cannot repair another. Use abstain whenever its boundary applies. "
        f"{retry}Return one schema-valid JSON object only."
    )
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=True, sort_keys=True)),
    )


@dataclass(frozen=True, slots=True)
class HostedPolicySetBatchAdapter:
    teacher: TeacherSpec
    backend: CompletionBackend

    def generate_policy_batch(self, request: PolicySetBatchGenerationRequest) -> str:
        if request.teacher_id != self.teacher.teacher_id or request.model_id != self.teacher.model_id:
            raise TeacherGenerationUnavailable(request.teacher_id, "adapter_identity_mismatch")
        reason_codes = tuple(code for policy in request.policies for code in policy.reason_codes.all())
        return self.backend.complete(
            CompletionRequest(
                model_id=request.model_id,
                messages=_messages(request),
                maximum_new_tokens=self.teacher.maximum_new_tokens,
                response_format=StructuredResponseFormat("json_object", reason_codes),
                temperature=self.teacher.temperature,
                top_p=self.teacher.top_p,
                reasoning_control=self.teacher.reasoning_control,
            )
        )


def _unavailable(teacher: TeacherSpec, units: tuple[EvaluationUnit, ...], policies: tuple[QualityPolicy, ...]) -> dict[str, tuple[TeacherVote, ...]]:
    return {
        unit.unit_id: tuple(
            TeacherVote(
                teacher_id=teacher.teacher_id,
                policy_id=policy.policy_id,
                decision=PolicyDecision.ABSTAIN,
                reason_codes=("teacher_generation_unavailable",),
            )
            for policy in policies
        )
        for unit in units
    }


def _evaluate_teacher_batch(
    adapter: PolicySetBatchAdapter,
    teacher: TeacherSpec,
    policies: tuple[QualityPolicy, ...],
    units: tuple[EvaluationUnit, ...],
    pass_index: Literal[1, 2],
    evidence_store: TeacherBatchEvidenceStoreProtocol | None,
) -> dict[str, tuple[TeacherVote, ...]]:
    if evidence_store is not None:
        cached = evidence_store.get(teacher, policies, units, pass_index)
        if cached is not None:
            print(
                json.dumps(
                    {
                        "quality_teacher": teacher.teacher_id,
                        "pass_index": pass_index,
                        "unit_count": len(units),
                        "status": "provider_cache_hit",
                    }
                ),
                flush=True,
            )
            return cached
    request = PolicySetBatchGenerationRequest(
        teacher_id=teacher.teacher_id,
        model_id=teacher.model_id,
        policies=policies,
        units=units,
        pass_index=pass_index,
        blind_run_id=uuid4().hex,
        schema_retry=False,
    )
    for retry in (False, True):
        started = perf_counter()
        try:
            raw = adapter.generate_policy_batch(request.model_copy(update={"schema_retry": retry}))
        except TeacherGenerationUnavailable as error:
            print(
                json.dumps(
                    {
                        "quality_teacher": teacher.teacher_id,
                        "pass_index": pass_index,
                        "schema_retry": retry,
                        "unit_count": len(units),
                        "status": "unavailable",
                        "reason": error.reason,
                        "elapsed_seconds": round(perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
            return _unavailable(teacher, units, policies)
        parsed = parse_policy_set_batch_response(raw, units, policies)
        if parsed is not None:
            votes_by_unit = {
                unit_id: tuple(
                    TeacherVote(
                        teacher_id=teacher.teacher_id,
                        policy_id=vote.policy_id,
                        decision=vote.decision,
                        reason_codes=vote.reason_codes,
                    )
                    for vote in votes
                )
                for unit_id, votes in parsed.items()
            }
            if evidence_store is not None:
                evidence_store.put(teacher, policies, units, pass_index, votes_by_unit)
            print(
                json.dumps(
                    {
                        "quality_teacher": teacher.teacher_id,
                        "pass_index": pass_index,
                        "schema_retry": retry,
                        "unit_count": len(units),
                        "status": "success",
                        "elapsed_seconds": round(perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
            return votes_by_unit
        print(
            json.dumps(
                {
                    "quality_teacher": teacher.teacher_id,
                    "pass_index": pass_index,
                    "schema_retry": retry,
                    "unit_count": len(units),
                    "status": "invalid_schema",
                    "elapsed_seconds": round(perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
    return {
        unit.unit_id: tuple(
            TeacherVote(
                teacher_id=teacher.teacher_id,
                policy_id=policy.policy_id,
                decision=PolicyDecision.ABSTAIN,
                reason_codes=("invalid_teacher_response_schema",),
            )
            for policy in policies
        )
        for unit in units
    }


def _run_pass(
    panel: TeacherPanel,
    adapters: Mapping[str, PolicySetBatchAdapter],
    units: tuple[EvaluationUnit, ...],
    pass_index: Literal[1, 2],
    evidence_store: TeacherBatchEvidenceStoreProtocol | None,
) -> tuple[dict[str, tuple[TeacherVote, ...]], ...]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        return tuple(
            executor.map(
                lambda teacher: _evaluate_teacher_batch(
                    adapters[teacher.teacher_id],
                    teacher,
                    panel.policies,
                    units,
                    pass_index,
                    evidence_store,
                ),
                panel.teachers,
            )
        )


def _needs_second(votes: tuple[TeacherVote, ...]) -> bool:
    counts = Counter(vote.decision for vote in votes)
    return decide_panel(votes, None) is PanelDecision.ABSTAIN and max(
        counts[PolicyDecision.PASS], counts[PolicyDecision.FAIL]
    ) == 2


def _teacher_batch_available(batch: Mapping[str, tuple[TeacherVote, ...]]) -> bool:
    return any(
        not NON_EVIDENCE_REASON_CODES.intersection(vote.reason_codes)
        for unit_votes in batch.values()
        for vote in unit_votes
    )


def evaluate_quality_units_batched(
    panel: TeacherPanel,
    adapters: Mapping[str, PolicySetBatchAdapter],
    units: tuple[EvaluationUnit, ...],
    *,
    evidence_store: TeacherBatchEvidenceStoreProtocol | None = None,
) -> tuple[UnitBatchResult, ...]:
    first = _run_pass(panel, adapters, units, 1, evidence_store)
    available = tuple(
        teacher.teacher_id
        for teacher, batch in zip(panel.teachers, first, strict=True)
        if _teacher_batch_available(batch)
    )
    if len(available) < 2:
        raise InsufficientTeacherAvailability(",".join(unit.unit_id for unit in units), len(available))
    needs_by_unit: dict[str, tuple[bool, ...]] = {}
    for unit in units:
        first_by_policy = tuple(tuple(batch[unit.unit_id][index] for batch in first) for index in range(4))
        needs_by_unit[unit.unit_id] = tuple(_needs_second(votes) for votes in first_by_policy)
    second_units = tuple(unit for unit in units if any(needs_by_unit[unit.unit_id]))
    second = _run_pass(panel, adapters, second_units, 2, evidence_store) if second_units else None
    unavailable = tuple(teacher.teacher_id for teacher in panel.teachers if teacher.teacher_id not in available)
    output: list[UnitBatchResult] = []
    for unit in units:
        policy_results: list[PanelPolicyResult] = []
        for index, policy in enumerate(panel.policies):
            if policy.policy_id == "q1_correctness_evidence" and unit.declared_verifier is not None:
                passed = unit.declared_verifier.status == "pass"
                policy_results.append(
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
            first_votes = tuple(batch[unit.unit_id][index] for batch in first)
            second_votes = (
                tuple(batch[unit.unit_id][index] for batch in second)
                if second is not None and needs_by_unit[unit.unit_id][index]
                else None
            )
            policy_results.append(
                PanelPolicyResult(
                    policy.policy_id,
                    decide_panel(first_votes, second_votes),
                    first_votes,
                    second_votes,
                )
            )
        output.append(
            UnitBatchResult(
                unit.unit_id,
                CombinedUnitResult(tuple(policy_results), available, unavailable),
            )
        )
    return tuple(output)
