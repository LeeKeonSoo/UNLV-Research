from __future__ import annotations

import json
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Iterable, Literal, Protocol, assert_never

from quality_teacher_panel import ReasoningControl, TeacherSpec
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
    temperature: float = 0.0
    top_p: float = 1.0
    reasoning_control: ReasoningControl = "none"


@dataclass(frozen=True, slots=True)
class StructuredResponseFormat:
    type: Literal["json_object"]
    allowed_reason_codes: tuple[str, ...]


class CompletionBackend(Protocol):
    def complete(self, request: CompletionRequest) -> str: ...


class ConcurrencyLimitedBackend:
    """Admit no more than the provider-specific number of concurrent requests."""

    __slots__ = ("_backend", "_limiter")

    def __init__(
        self,
        *,
        backend: CompletionBackend,
        maximum_concurrent_requests: int,
    ) -> None:
        if maximum_concurrent_requests < 1:
            raise TeacherAdapterContractError(
                teacher_id="concurrency-limiter",
                detail="maximum_concurrent_requests must be positive",
            )
        self._backend = backend
        self._limiter = BoundedSemaphore(maximum_concurrent_requests)

    def complete(self, request: CompletionRequest) -> str:
        with self._limiter:
            return self._backend.complete(request)


def build_reasoning_extra_body(
    control: ReasoningControl,
) -> dict[str, object]:
    match control:
        case "none":
            return {}
        case "enable_thinking_false":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        case "thinking_false":
            return {"chat_template_kwargs": {"thinking": False}}
        case "reasoning_effort_none":
            return {"reasoning_effort": "none"}
        case unreachable:
            assert_never(unreachable)


def collect_stream_content(chunks: Iterable[object]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None)
        if content:
            parts.append(str(content))
    return "".join(parts).strip()


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
        "response_contract": {
            "required_object_keys": ["decision", "reason_codes"],
            "allowed_decisions": ["pass", "fail", "abstain"],
            "allowed_reason_codes_by_decision": {
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
        "when the policy boundary requires unavailable evidence. The output object must contain "
        'exactly two keys named "decision" and "reason_codes". Do not output the allowed-code '
        f"mapping or any schema description. {retry_instruction}"
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
                temperature=self.teacher.temperature,
                top_p=self.teacher.top_p,
                reasoning_control=self.teacher.reasoning_control,
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
        from httpx import TimeoutException
        from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

        try:
            completion = self._client.chat.completions.create(
                model=request.model_id,
                messages=[
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.maximum_new_tokens,
                response_format={"type": request.response_format.type},
                extra_body=build_reasoning_extra_body(request.reasoning_control),
                stream=True,
            )
            content = collect_stream_content(completion)
        except APITimeoutError as error:
            raise TeacherGenerationUnavailable(request.model_id, "read_timeout") from error
        except TimeoutException as error:
            raise TeacherGenerationUnavailable(request.model_id, "read_timeout") from error
        except APIConnectionError as error:
            raise TeacherGenerationUnavailable(request.model_id, "connection_error") from error
        except APIStatusError as error:
            raise TeacherGenerationUnavailable(
                request.model_id,
                f"http_status_{error.status_code}",
            ) from error
        except APIError as error:
            raise TeacherGenerationUnavailable(
                request.model_id,
                f"api_error_{type(error).__name__.lower()}",
            ) from error
        if not content:
            raise TeacherAdapterContractError(
                teacher_id=request.model_id,
                detail="endpoint returned no textual completion",
            )
        return content
