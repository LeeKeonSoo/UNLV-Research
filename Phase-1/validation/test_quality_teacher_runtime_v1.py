#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import PanelDecision, load_teacher_panel
from quality_teacher_runtime import (
    EvaluationUnit,
    TeacherGenerationRequest,
    evaluate_quality_unit,
    evaluate_panel_policy,
    evaluate_teacher,
    resolve_quality_gate,
)


CONFIG = ROOT / "configs" / "quality_teacher_panel_v1.json"


class ScriptedAdapter:
    """In-memory adapter that records the machine-consumed request contract."""

    def __init__(self, responses: tuple[str, ...]) -> None:
        self._responses = deque(responses)
        self.requests: list[TeacherGenerationRequest] = []

    def generate(self, request: TeacherGenerationRequest) -> str:
        self.requests.append(request)
        return self._responses.popleft()


def _unit() -> EvaluationUnit:
    return EvaluationUnit(
        unit_id="public-fixture-001",
        text="A square has four equal sides.",
        declared_context="English educational prose.",
        attached_evidence=(),
    )


def test_teacher_runtime_stops_after_schema_valid_first_response() -> None:
    # Given: a teacher returns a schema-valid decision immediately.
    panel = load_teacher_panel(CONFIG)
    adapter = ScriptedAdapter(('{"decision":"pass","reason_codes":["observable_relation"]}',))

    # When: one teacher evaluates one policy.
    vote = evaluate_teacher(adapter, panel.teachers[0], panel.policies[3], _unit(), pass_index=1)

    # Then: the valid response is accepted without a retry.
    assert vote.decision.value == "pass"
    assert len(adapter.requests) == 1
    assert adapter.requests[0].schema_retry is False


def test_teacher_runtime_retries_schema_once_without_changing_policy() -> None:
    # Given: a teacher first violates the enum and then follows the schema.
    panel = load_teacher_panel(CONFIG)
    adapter = ScriptedAdapter(
        (
            '{"decision":"ACCEPT","reason_codes":["observable_relation"]}',
            '{"decision":"pass","reason_codes":["observable_relation"]}',
        )
    )

    # When: the teacher response boundary is executed.
    vote = evaluate_teacher(adapter, panel.teachers[0], panel.policies[3], _unit(), pass_index=1)

    # Then: exactly one schema-only retry preserves the evaluation identity.
    assert vote.decision.value == "pass"
    assert len(adapter.requests) == 2
    first, retry = adapter.requests
    assert (retry.teacher_id, retry.policy_id, retry.unit_id, retry.pass_index) == (
        first.teacher_id,
        first.policy_id,
        first.unit_id,
        first.pass_index,
    )
    assert retry.schema_retry is True


def test_teacher_runtime_abstains_after_two_invalid_responses() -> None:
    # Given: both the initial output and the single retry are invalid.
    panel = load_teacher_panel(CONFIG)
    adapter = ScriptedAdapter(("```json\n{}\n```", '{"decision":false,"reason_codes":[]}'))

    # When: the teacher response boundary is executed.
    vote = evaluate_teacher(adapter, panel.teachers[1], panel.policies[2], _unit(), pass_index=1)

    # Then: malformed output cannot become quality evidence.
    assert vote.decision.value == "abstain"
    assert vote.reason_codes == ("invalid_teacher_response_schema",)
    assert len(adapter.requests) == 2


def test_panel_runtime_repeats_nonunanimous_majority_blinded() -> None:
    # Given: first-pass 2/3 agreement remains stable on a second blinded pass.
    panel = load_teacher_panel(CONFIG)
    adapters = {
        panel.teachers[0].teacher_id: ScriptedAdapter(
            (
                '{"decision":"fail","reason_codes":["no_payload"]}',
                '{"decision":"fail","reason_codes":["no_payload"]}',
            )
        ),
        panel.teachers[1].teacher_id: ScriptedAdapter(
            (
                '{"decision":"fail","reason_codes":["no_payload"]}',
                '{"decision":"fail","reason_codes":["no_payload"]}',
            )
        ),
        panel.teachers[2].teacher_id: ScriptedAdapter(
            (
                '{"decision":"pass","reason_codes":["specialized_payload"]}',
                '{"decision":"abstain","reason_codes":["insufficient_context"]}',
            )
        ),
    }

    # When: the complete Q3 panel policy is evaluated.
    result = evaluate_panel_policy(panel, adapters, panel.policies[2], _unit())

    # Then: the stable repeated majority is accepted and every teacher ran twice.
    assert result.decision is PanelDecision.FAIL
    assert result.second_pass is not None
    assert all(len(adapter.requests) == 2 for adapter in adapters.values())
    assert all(adapter.requests[1].pass_index == 2 for adapter in adapters.values())
    assert len({adapter.requests[1].blind_run_id for adapter in adapters.values()}) == 1


def test_panel_runtime_repeats_majority_with_one_abstention() -> None:
    # Given: two teachers agree while the third abstains on both passes.
    panel = load_teacher_panel(CONFIG)
    adapters = {
        panel.teachers[0].teacher_id: ScriptedAdapter(
            (
                '{"decision":"pass","reason_codes":["coherent_unit"]}',
                '{"decision":"pass","reason_codes":["coherent_unit"]}',
            )
        ),
        panel.teachers[1].teacher_id: ScriptedAdapter(
            (
                '{"decision":"pass","reason_codes":["coherent_unit"]}',
                '{"decision":"pass","reason_codes":["coherent_unit"]}',
            )
        ),
        panel.teachers[2].teacher_id: ScriptedAdapter(
            (
                '{"decision":"abstain","reason_codes":["insufficient_context"]}',
                '{"decision":"abstain","reason_codes":["insufficient_context"]}',
            )
        ),
    }

    # When: the complete Q2 panel policy is evaluated.
    result = evaluate_panel_policy(panel, adapters, panel.policies[1], _unit())

    # Then: the nonunanimous majority is repeated before acceptance.
    assert result.decision is PanelDecision.PASS
    assert result.second_pass is not None
    assert all(len(adapter.requests) == 2 for adapter in adapters.values())


def test_quality_runtime_evaluates_q1_to_q4_as_independent_gates() -> None:
    # Given: every teacher returns one valid pass for each of four policies.
    panel = load_teacher_panel(CONFIG)
    response = '{"decision":"pass","reason_codes":["observable_policy_evidence"]}'
    adapters = {
        teacher.teacher_id: ScriptedAdapter((response, response, response, response))
        for teacher in panel.teachers
    }

    # When: one unit is evaluated by the complete Quality candidate.
    results = evaluate_quality_unit(panel, adapters, _unit())

    # Then: Q1-Q4 remain four independent decisions rather than one scalar score.
    assert tuple(result.policy_id for result in results) == tuple(
        policy.policy_id for policy in panel.policies
    )
    assert all(result.decision is PanelDecision.PASS for result in results)
    assert all(len(adapter.requests) == 4 for adapter in adapters.values())


def test_quality_gate_fails_when_any_independent_policy_fails() -> None:
    # Given: Q1-Q3 pass while Q4 has a stable unanimous failure.
    panel = load_teacher_panel(CONFIG)
    pass_raw = '{"decision":"pass","reason_codes":["observable_policy_evidence"]}'
    fail_raw = '{"decision":"fail","reason_codes":["no_recoverable_relation"]}'
    adapters = {
        teacher.teacher_id: ScriptedAdapter((pass_raw, pass_raw, pass_raw, fail_raw))
        for teacher in panel.teachers
    }

    # When: the four policy results are combined as independent fail gates.
    decision = resolve_quality_gate(evaluate_quality_unit(panel, adapters, _unit()))

    # Then: one failed policy is sufficient for the candidate Quality failure.
    assert decision is PanelDecision.FAIL


if __name__ == "__main__":
    test_teacher_runtime_stops_after_schema_valid_first_response()
    test_teacher_runtime_retries_schema_once_without_changing_policy()
    test_teacher_runtime_abstains_after_two_invalid_responses()
    test_panel_runtime_repeats_nonunanimous_majority_blinded()
    test_panel_runtime_repeats_majority_with_one_abstention()
    test_quality_runtime_evaluates_q1_to_q4_as_independent_gates()
    test_quality_gate_fails_when_any_independent_policy_fails()
    print("[quality-teacher-runtime-v1] adapter, retry, and blinded consensus: pass")
