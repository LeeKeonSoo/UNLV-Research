#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_adapters import (
    CompletionRequest,
    TeacherAdapterContractError,
    TeacherModelAdapter,
    build_teacher_messages,
    collect_stream_content,
)
from quality_teacher_panel import load_teacher_panel
from quality_teacher_runtime import EvaluationUnit, TeacherGenerationRequest
from quality_teacher_local import extract_chat_input_ids


CONFIG = ROOT / "configs" / "quality_teacher_panel_v1.json"


class RecordingBackend:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> str:
        self.requests.append(request)
        return self.response


def _request(*, teacher_id: str, model_id: str, schema_retry: bool) -> TeacherGenerationRequest:
    return TeacherGenerationRequest(
        teacher_id=teacher_id,
        model_id=model_id,
        policy_id="q1_correctness_evidence",
        policy_name="Correctness Evidence",
        policy_question="Is correctness supported by observable evidence?",
        fail_boundary="Fail only on a reproducible contradiction.",
        abstain_boundary="Abstain when required evidence is unavailable.",
        pass_reason_codes=("observable_correctness_evidence",),
        fail_reason_codes=("reproducible_contradiction",),
        abstain_reason_codes=("external_knowledge_required",),
        unit_id="public-fixture-001",
        unit_text="Two plus two equals four.",
        declared_context="English educational prose.",
        attached_evidence=("2 + 2 = 4",),
        pass_index=1,
        blind_run_id="blind-001",
        schema_retry=schema_retry,
    )


def test_prompt_builder_emits_machine_readable_policy_and_schema_contract() -> None:
    # Given: a typed first-pass teacher request.
    request = _request(teacher_id="teacher-a", model_id="model-a", schema_retry=False)

    # When: the adapter builds messages for a backend.
    messages = build_teacher_messages(request)
    payload = json.loads(messages[1].content)

    # Then: routing, evidence, and output schema remain machine readable.
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert payload["policy"]["policy_id"] == "q1_correctness_evidence"
    assert payload["unit"]["unit_id"] == "public-fixture-001"
    assert payload["unit"]["attached_evidence"] == ["2 + 2 = 4"]
    assert payload["execution"]["pass_index"] == 1
    assert payload["execution"]["blind_run_id"] == "blind-001"
    assert payload["execution"]["schema_retry"] is False
    assert payload["response_contract"]["required_object_keys"] == ["decision", "reason_codes"]
    assert payload["response_contract"]["allowed_decisions"] == ["pass", "fail", "abstain"]
    assert payload["response_contract"]["allowed_reason_codes_by_decision"]["fail"] == [
        "reproducible_contradiction"
    ]
    assert "Do not output the allowed-code mapping" in messages[0].content


def test_teacher_model_adapter_routes_frozen_model_and_returns_raw_response() -> None:
    # Given: an adapter bound to one frozen hosted teacher and a recording backend.
    teacher = load_teacher_panel(CONFIG).teachers[0]
    backend = RecordingBackend('{"decision":"pass","reason_codes":["observable_evidence"]}')
    adapter = TeacherModelAdapter(teacher=teacher, backend=backend, maximum_new_tokens=192)

    # When: the teacher generates one response.
    raw = adapter.generate(
        _request(teacher_id=teacher.teacher_id, model_id=teacher.model_id, schema_retry=False)
    )

    # Then: the frozen model identity and bounded output budget reach the backend.
    assert raw == backend.response
    assert len(backend.requests) == 1
    assert backend.requests[0].model_id == teacher.model_id
    assert backend.requests[0].maximum_new_tokens == 192
    assert backend.requests[0].response_format.type == "json_object"
    assert backend.requests[0].response_format.allowed_reason_codes == (
        "observable_correctness_evidence",
        "reproducible_contradiction",
        "external_knowledge_required",
    )


def test_teacher_model_adapter_rejects_cross_teacher_dispatch() -> None:
    # Given: an adapter bound to one teacher but a request naming another teacher.
    teacher = load_teacher_panel(CONFIG).teachers[0]
    backend = RecordingBackend('{"decision":"pass","reason_codes":["observable_evidence"]}')
    adapter = TeacherModelAdapter(teacher=teacher, backend=backend, maximum_new_tokens=192)
    request = _request(teacher_id="different-teacher", model_id=teacher.model_id, schema_retry=False)

    # When/Then: identity drift is rejected before any model call.
    try:
        adapter.generate(request)
    except TeacherAdapterContractError as error:
        assert error.teacher_id == "different-teacher"
    else:
        raise AssertionError("Cross-teacher dispatch must be rejected")
    assert backend.requests == []


def test_local_chat_template_extracts_input_ids_from_batch_encoding() -> None:
    # Given: Qwen3.5 returns a BatchEncoding-shaped mapping instead of a raw tensor.
    expected_ids = (101, 102, 103)
    encoded = {"input_ids": expected_ids, "attention_mask": (1, 1, 1)}

    # When: the local adapter resolves the generation input.
    input_ids = extract_chat_input_ids(encoded)

    # Then: only the model-consumed input_ids field is forwarded.
    assert input_ids is expected_ids


def test_stream_collector_ignores_empty_transport_events() -> None:
    chunks = (
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='{"decision":'))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='"pass"}'))]),
    )

    assert collect_stream_content(chunks) == '{"decision":"pass"}'


if __name__ == "__main__":
    test_prompt_builder_emits_machine_readable_policy_and_schema_contract()
    test_teacher_model_adapter_routes_frozen_model_and_returns_raw_response()
    test_teacher_model_adapter_rejects_cross_teacher_dispatch()
    test_local_chat_template_extracts_input_ids_from_batch_encoding()
    test_stream_collector_ignores_empty_transport_events()
    print("[quality-teacher-adapters-v1] prompt and model routing contract: pass")
