from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import PanelDecision, PolicyDecision, load_teacher_panel
from quality_teacher_runtime import EvaluationUnit, TeacherGenerationUnavailable
from quality_teacher_unit_runtime import (
    InsufficientTeacherAvailability,
    PolicySetGenerationRequest,
    evaluate_quality_unit_combined,
    parse_policy_set_response,
)
from quality_teacher_batch_runtime import (
    PolicySetBatchGenerationRequest,
    evaluate_quality_units_batched,
    parse_policy_set_batch_response,
)


PANEL = ROOT / "configs" / "quality_teacher_panel_v2.json"


def _response(decision_by_policy: dict[str, str]) -> str:
    reason_by_policy = {
        "q1_correctness_evidence": {
            "pass": "observable_correctness_evidence",
            "fail": "locally_checkable_incorrect_result",
        },
        "q2_semantic_coherence": {
            "pass": "recoverable_semantic_unit",
            "fail": "internal_semantic_contradiction",
        },
        "q3_substantive_payload": {
            "pass": "substantive_payload_present",
            "fail": "boilerplate_only",
        },
        "q4_learnable_relations": {
            "pass": "recoverable_relation_present",
            "fail": "fragment_set_without_relation",
        },
    }
    return json.dumps(
        {
            "policies": [
                {
                    "policy_id": policy_id,
                    "decision": decision,
                    "reason_codes": [reason_by_policy[policy_id][decision]],
                }
                for policy_id, decision in decision_by_policy.items()
            ]
        }
    )


class ControlledAdapter:
    def __init__(self, teacher_id: str, *, unavailable: bool = False) -> None:
        self.teacher_id = teacher_id
        self.unavailable = unavailable
        self.calls: list[PolicySetGenerationRequest] = []

    def generate_policy_set(self, request: PolicySetGenerationRequest) -> str:
        self.calls.append(request)
        if self.unavailable:
            raise TeacherGenerationUnavailable(self.teacher_id, "controlled_unavailable")
        decisions = {policy.policy_id: "pass" for policy in request.policies}
        decisions["q3_substantive_payload"] = "fail"
        return _response(decisions)


def _batch_response(unit_ids: tuple[str, ...]) -> str:
    decisions = {
        "q1_correctness_evidence": "pass",
        "q2_semantic_coherence": "pass",
        "q3_substantive_payload": "fail",
        "q4_learnable_relations": "pass",
    }
    return json.dumps(
        {
            "units": [
                {
                    "unit_id": unit_id,
                    **json.loads(_response(decisions)),
                }
                for unit_id in unit_ids
            ]
        }
    )


class ControlledBatchAdapter:
    def __init__(
        self,
        teacher_id: str,
        *,
        unavailable: bool = False,
        invalid_schema: bool = False,
    ) -> None:
        self.teacher_id = teacher_id
        self.unavailable = unavailable
        self.invalid_schema = invalid_schema
        self.calls: list[PolicySetBatchGenerationRequest] = []

    def generate_policy_batch(self, request: PolicySetBatchGenerationRequest) -> str:
        self.calls.append(request)
        if self.unavailable:
            raise TeacherGenerationUnavailable(self.teacher_id, "controlled_unavailable")
        if self.invalid_schema:
            return "{}"
        return _batch_response(tuple(unit.unit_id for unit in request.units))


def test_combined_response_requires_each_policy_exactly_once() -> None:
    panel = load_teacher_panel(PANEL)
    complete = _response({policy.policy_id: "pass" for policy in panel.policies})
    parsed = parse_policy_set_response(complete, panel.policies)
    assert parsed is not None
    assert {vote.policy_id for vote in parsed} == {
        "q1_correctness_evidence",
        "q2_semantic_coherence",
        "q3_substantive_payload",
        "q4_learnable_relations",
    }

    missing_q4 = _response(
        {policy.policy_id: "pass" for policy in panel.policies if policy.policy_id != "q4_learnable_relations"}
    )
    assert parse_policy_set_response(missing_q4, panel.policies) is None


def test_two_available_teachers_can_create_only_stable_majority_fail() -> None:
    panel = load_teacher_panel(PANEL)
    adapters = {
        teacher.teacher_id: ControlledAdapter(
            teacher.teacher_id,
            unavailable=index == 2,
        )
        for index, teacher in enumerate(panel.teachers)
    }
    unit = EvaluationUnit(
        unit_id="fixture",
        text="generated boilerplate fixture",
        declared_context=None,
        attached_evidence=(),
    )
    result = evaluate_quality_unit_combined(panel, adapters, unit)
    by_policy = {policy.policy_id: policy for policy in result.policy_results}
    assert result.available_teacher_ids == tuple(teacher.teacher_id for teacher in panel.teachers[:2])
    assert by_policy["q3_substantive_payload"].decision is PanelDecision.FAIL
    assert by_policy["q3_substantive_payload"].second_pass is not None
    assert all(
        vote.decision is PolicyDecision.FAIL
        for vote in by_policy["q3_substantive_payload"].first_pass[:2]
    )
    assert by_policy["q1_correctness_evidence"].decision is PanelDecision.PASS


def test_one_available_teacher_cannot_produce_cacheable_evidence() -> None:
    panel = load_teacher_panel(PANEL)
    adapters = {
        teacher.teacher_id: ControlledAdapter(
            teacher.teacher_id,
            unavailable=index != 0,
        )
        for index, teacher in enumerate(panel.teachers)
    }
    unit = EvaluationUnit(
        unit_id="fixture",
        text="payload",
        declared_context=None,
        attached_evidence=(),
    )
    try:
        evaluate_quality_unit_combined(panel, adapters, unit)
    except InsufficientTeacherAvailability as error:
        assert error.available_teachers == 1
    else:
        raise AssertionError("fewer than two available teachers must not yield evidence")


def test_batch_transport_preserves_unit_and_policy_matrix() -> None:
    panel = load_teacher_panel(PANEL)
    units = tuple(
        EvaluationUnit(
            unit_id=f"unit-{index}",
            text=f"payload {index}",
            declared_context=None,
            attached_evidence=(),
        )
        for index in range(2)
    )
    raw = _batch_response(tuple(unit.unit_id for unit in units))
    parsed = parse_policy_set_batch_response(raw, units, panel.policies)
    assert parsed is not None
    assert set(parsed) == {"unit-0", "unit-1"}

    missing = json.loads(raw)
    missing["units"].pop()
    assert parse_policy_set_batch_response(json.dumps(missing), units, panel.policies) is None

    adapters = {
        teacher.teacher_id: ControlledBatchAdapter(
            teacher.teacher_id,
            unavailable=index == 2,
        )
        for index, teacher in enumerate(panel.teachers)
    }
    results = evaluate_quality_units_batched(panel, adapters, units)
    assert tuple(result.unit_id for result in results) == ("unit-0", "unit-1")
    assert all(
        next(
            policy
            for policy in result.evidence.policy_results
            if policy.policy_id == "q3_substantive_payload"
        ).decision
        is PanelDecision.FAIL
        for result in results
    )
    assert all(len(adapter.calls) == 2 for adapter in adapters.values())


def test_batch_invalid_schema_does_not_count_as_teacher_availability() -> None:
    panel = load_teacher_panel(PANEL)
    unit = EvaluationUnit(
        unit_id="invalid-schema-availability",
        text="A valid synthetic payload.",
        declared_context=None,
        attached_evidence=(),
    )
    adapters = {
        teacher.teacher_id: ControlledBatchAdapter(
            teacher.teacher_id,
            invalid_schema=index != 0,
        )
        for index, teacher in enumerate(panel.teachers)
    }

    try:
        evaluate_quality_units_batched(panel, adapters, (unit,))
    except InsufficientTeacherAvailability as error:
        assert error.available_teachers == 1
    else:
        raise AssertionError("schema-invalid teachers must not satisfy the availability gate")


if __name__ == "__main__":
    test_combined_response_requires_each_policy_exactly_once()
    test_two_available_teachers_can_create_only_stable_majority_fail()
    test_one_available_teacher_cannot_produce_cacheable_evidence()
    test_batch_transport_preserves_unit_and_policy_matrix()
    test_batch_invalid_schema_does_not_count_as_teacher_availability()
    print("[quality-teacher-unit-runtime-v1] combined independent policy contract: pass")
