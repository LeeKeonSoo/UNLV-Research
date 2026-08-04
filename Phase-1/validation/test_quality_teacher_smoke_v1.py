#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_runtime import TeacherGenerationRequest, TeacherGenerationUnavailable
from quality_teacher_smoke import AuditedAdapter


class ScriptedDelegate:
    def generate(self, request: TeacherGenerationRequest) -> str:
        return '{"decision":"pass","reason_codes":["substantive_payload_present"]}'


class UnavailableDelegate:
    def generate(self, request: TeacherGenerationRequest) -> str:
        raise TeacherGenerationUnavailable(request.teacher_id, "read_timeout")


def _request() -> TeacherGenerationRequest:
    return TeacherGenerationRequest(
        teacher_id="teacher-a",
        model_id="model-a",
        policy_id="q3_substantive_payload",
        policy_name="Substantive Payload",
        policy_question="Is substantive payload present?",
        fail_boundary="Fail only without residual payload.",
        abstain_boundary="Abstain when specialized payload is uncertain.",
        pass_reason_codes=("substantive_payload_present",),
        fail_reason_codes=("no_substantive_residual",),
        abstain_reason_codes=("specialized_payload_uncertain",),
        unit_id="fixture-001",
        unit_text="A rectangle has four sides.",
        declared_context="English synthetic prose.",
        attached_evidence=(),
        pass_index=1,
        blind_run_id="blind-001",
        schema_retry=False,
    )


def test_audited_adapter_records_generation_latency_without_raw_text() -> None:
    # Given: a deterministic clock and one schema-valid model response.
    ticks = deque((10.0, 10.25))
    adapter = AuditedAdapter(delegate=ScriptedDelegate(), clock=ticks.popleft)

    # When: one teacher generation is audited.
    raw = adapter.generate(_request())

    # Then: latency and response identity are recorded without storing response text.
    assert raw.startswith("{")
    assert adapter.traces[0]["elapsed_milliseconds"] == 250
    assert "raw_response" not in adapter.traces[0]


def test_audited_adapter_records_failed_generation_without_swallowing_error() -> None:
    # Given: one hosted teacher exceeds its frozen request timeout.
    ticks = deque((20.0, 20.75))
    adapter = AuditedAdapter(delegate=UnavailableDelegate(), clock=ticks.popleft)

    # When: the audited call fails with a typed availability error.
    try:
        adapter.generate(_request())
    except TeacherGenerationUnavailable as error:
        assert error.reason == "read_timeout"
    else:
        raise AssertionError("The audit wrapper must preserve generation failures")

    # Then: the failure is observable without storing prompt or response content.
    assert adapter.traces == [
        {
            "teacher_id": "teacher-a",
            "policy_id": "q3_substantive_payload",
            "pass_index": 1,
            "schema_retry": False,
            "elapsed_milliseconds": 750,
            "status": "unavailable",
            "error_reason": "read_timeout",
        }
    ]


if __name__ == "__main__":
    test_audited_adapter_records_generation_latency_without_raw_text()
    test_audited_adapter_records_failed_generation_without_swallowing_error()
    print("[quality-teacher-smoke-v1] latency and response-hash audit: pass")
