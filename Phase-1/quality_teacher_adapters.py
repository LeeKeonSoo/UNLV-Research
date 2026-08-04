from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from quality_teacher_panel import TeacherSpec
from quality_teacher_runtime import TeacherGenerationRequest


@dataclass(frozen=True, slots=True)
class TeacherAdapterContractError(RuntimeError):
    teacher_id: str
    detail: str

    def __str__(self) -> str:
        return f"Teacher adapter contract failed for {self.teacher_id}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user"]
    content: str


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model_id: str
    messages: tuple[ChatMessage, ChatMessage]
    maximum_new_tokens: int


class CompletionBackend(Protocol):
    def complete(self, request: CompletionRequest) -> str: ...


def build_teacher_messages(
    request: TeacherGenerationRequest,
) -> tuple[ChatMessage, ChatMessage]:
    payload = {
        "policy": {
            "policy_id": request.policy_id,
            "name": request.policy_name,
            "question": request.policy_question,
            "fail_boundary": request.fail_boundary,
            "abstain_boundary": request.abstain_boundary,
        },
        "unit": {
            "unit_id": request.unit_id,
            "text": request.unit_text,
            "declared_context": request.declared_context,
            "attached_evidence": list(request.attached_evidence),
        },
        "execution": {
            "pass_index": request.pass_index,
            "blind_run_id": request.blind_run_id,
            "schema_retry": request.schema_retry,
            "prior_panel_votes_available": False,
        },
        "response_schema": {
            "decision": ["pass", "fail", "abstain"],
            "reason_codes": "array_of_1_to_8_lower_snake_case_strings_maximum_64_characters",
            "additional_properties": False,
        },
    }
    retry_instruction = (
        "The previous response violated the response schema. Re-evaluate independently and return "
        "one schema-valid JSON object only."
        if request.schema_retry
        else "Return one schema-valid JSON object only."
    )
    system = (
        "You are one independent evaluator in a data-curation qualification panel. Evaluate only "
        "the supplied policy and observable unit evidence. Never infer other panel votes. Use abstain "
        f"when the policy boundary requires unavailable evidence. {retry_instruction}"
    )
    return (
        ChatMessage(role="system", content=system),
        ChatMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        ),
    )


@dataclass(frozen=True, slots=True)
class TeacherModelAdapter:
    teacher: TeacherSpec
    backend: CompletionBackend
    maximum_new_tokens: int

    def __post_init__(self) -> None:
        if self.maximum_new_tokens <= 0:
            raise TeacherAdapterContractError(
                teacher_id=self.teacher.teacher_id,
                detail="maximum_new_tokens must be positive",
            )

    def generate(self, request: TeacherGenerationRequest) -> str:
        if request.teacher_id != self.teacher.teacher_id or request.model_id != self.teacher.model_id:
            raise TeacherAdapterContractError(
                teacher_id=request.teacher_id,
                detail="request identity does not match the frozen adapter teacher",
            )
        return self.backend.complete(
            CompletionRequest(
                model_id=self.teacher.model_id,
                messages=build_teacher_messages(request),
                maximum_new_tokens=self.maximum_new_tokens,
            )
        )


class NvidiaBuildBackend:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        if not api_key:
            raise TeacherAdapterContractError(
                teacher_id="nvidia-build",
                detail="API key is empty",
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=2,
            timeout=120.0,
        )

    def complete(self, request: CompletionRequest) -> str:
        completion = self._client.chat.completions.create(
            model=request.model_id,
            messages=[
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            temperature=0.0,
            max_tokens=request.maximum_new_tokens,
        )
        content = completion.choices[0].message.content
        if content is None or not content.strip():
            raise TeacherAdapterContractError(
                teacher_id=request.model_id,
                detail="endpoint returned no textual completion",
            )
        return content.strip()
