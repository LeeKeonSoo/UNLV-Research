from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from quality_teacher_panel import PolicyDecision, QualityPolicy, TeacherVote


ReasonCode = Annotated[
    str,
    StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"),
]


class TeacherResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: PolicyDecision
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1, max_length=8)


@dataclass(frozen=True, slots=True)
class TeacherResponseAttempt:
    teacher_id: str
    policy: QualityPolicy
    first_raw: str
    retry_raw: str | None


def parse_teacher_response(raw: str) -> TeacherResponsePayload | None:
    try:
        payload = json.loads(raw)
        return TeacherResponsePayload.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        return None


def _parse_policy_response(raw: str, policy: QualityPolicy) -> TeacherResponsePayload | None:
    payload = parse_teacher_response(raw)
    if payload is None:
        return None
    allowed = set(policy.reason_codes.for_decision(payload.decision))
    return payload if set(payload.reason_codes) <= allowed else None


def resolve_teacher_response(attempt: TeacherResponseAttempt) -> TeacherVote:
    first = _parse_policy_response(attempt.first_raw, attempt.policy)
    if first is not None:
        return TeacherVote(
            teacher_id=attempt.teacher_id,
            policy_id=attempt.policy.policy_id,
            decision=first.decision,
            reason_codes=first.reason_codes,
        )
    retry = (
        _parse_policy_response(attempt.retry_raw, attempt.policy)
        if attempt.retry_raw is not None
        else None
    )
    if retry is not None:
        return TeacherVote(
            teacher_id=attempt.teacher_id,
            policy_id=attempt.policy.policy_id,
            decision=retry.decision,
            reason_codes=retry.reason_codes,
        )
    return TeacherVote(
        teacher_id=attempt.teacher_id,
        policy_id=attempt.policy.policy_id,
        decision=PolicyDecision.ABSTAIN,
        reason_codes=("invalid_teacher_response_schema",),
    )
