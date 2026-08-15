#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_adapters import (
    ChatMessage,
    CompletionRequest,
    StructuredResponseFormat,
    TeacherAdapterContractError,
)
from quality_teacher_openai import OpenAIResponsesBackend


class RecordingResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


def _request() -> CompletionRequest:
    return CompletionRequest(
        model_id="gpt-5.6-luna",
        messages=(
            ChatMessage(role="system", content="Return one JSON object."),
            ChatMessage(role="user", content='{"units":[]}'),
        ),
        maximum_new_tokens=2048,
        response_format=StructuredResponseFormat(
            type="json_object",
            allowed_reason_codes=("substantive_payload_present",),
        ),
    )


def test_openai_backend_uses_responses_api_with_bounded_json_contract() -> None:
    responses = RecordingResponses('{"units":[]}')
    client = SimpleNamespace(responses=responses)
    backend = OpenAIResponsesBackend(
        api_key="test-key",
        timeout_seconds=90,
        maximum_transport_retries=1,
        reasoning_effort="low",
        client_factory=lambda **_: client,
    )

    raw = backend.complete(_request())

    assert raw == '{"units":[]}'
    assert len(responses.requests) == 1
    request = responses.requests[0]
    assert request["model"] == "gpt-5.6-luna"
    assert request["instructions"] == "Return one JSON object."
    assert request["input"] == '{"units":[]}'
    assert request["max_output_tokens"] == 2048
    assert request["reasoning"] == {"effort": "low"}
    assert request["text"] == {
        "format": {"type": "json_object"},
        "verbosity": "low",
    }
    assert request["store"] is False


def test_openai_backend_rejects_empty_output() -> None:
    client = SimpleNamespace(responses=RecordingResponses("   "))
    backend = OpenAIResponsesBackend(
        api_key="test-key",
        timeout_seconds=90,
        maximum_transport_retries=1,
        reasoning_effort="low",
        client_factory=lambda **_: client,
    )

    try:
        backend.complete(_request())
    except TeacherAdapterContractError as error:
        assert error.teacher_id == "gpt-5.6-luna"
        assert error.detail == "endpoint returned no textual completion"
    else:
        raise AssertionError("An empty Luna response must be rejected")


if __name__ == "__main__":
    test_openai_backend_uses_responses_api_with_bounded_json_contract()
    test_openai_backend_rejects_empty_output()
    print("[quality-teacher-openai-backend-v1] Responses API contract: pass")
