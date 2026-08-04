from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from quality_teacher_panel import TeacherSpec
from quality_teacher_runtime import TeacherGenerationRequest, TeacherGenerationUnavailable


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
    response_format: StructuredResponseFormat


@dataclass(frozen=True, slots=True)
class StructuredResponseFormat:
    type: Literal["json_object"]
    allowed_reason_codes: tuple[str, ...]


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
            "reason_codes_by_decision": {
                "pass": list(request.pass_reason_codes),
                "fail": list(request.fail_reason_codes),
                "abstain": list(request.abstain_reason_codes),
            },
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
                response_format=StructuredResponseFormat(
                    type="json_object",
                    allowed_reason_codes=(
                        request.pass_reason_codes
                        + request.fail_reason_codes
                        + request.abstain_reason_codes
                    ),
                ),
            )
        )


class NvidiaBuildBackend:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        maximum_transport_retries: int,
    ) -> None:
        if not api_key:
            raise TeacherAdapterContractError(
                teacher_id="nvidia-build",
                detail="API key is empty",
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            max_retries=maximum_transport_retries,
            timeout=float(timeout_seconds),
        )

    def complete(self, request: CompletionRequest) -> str:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        try:
            completion = self._client.chat.completions.create(
                model=request.model_id,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                temperature=0.0,
                max_tokens=request.maximum_new_tokens,
                response_format={"type": request.response_format.type},
            )
        except APITimeoutError as error:
            raise TeacherGenerationUnavailable(request.model_id, "read_timeout") from error
        except APIConnectionError as error:
            raise TeacherGenerationUnavailable(request.model_id, "connection_error") from error
        except APIStatusError as error:
            raise TeacherGenerationUnavailable(
                request.model_id,
                f"http_status_{error.status_code}",
            ) from error
        content = completion.choices[0].message.content
        if content is None or not content.strip():
            raise TeacherAdapterContractError(
                teacher_id=request.model_id,
                detail="endpoint returned no textual completion",
            )
        return content.strip()
