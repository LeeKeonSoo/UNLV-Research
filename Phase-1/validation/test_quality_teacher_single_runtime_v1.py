from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import (
    PanelDecision,
    PolicyDecision,
    TeacherVote,
    decide_single_teacher,
    load_teacher_panel,
)
from quality_teacher_batch_runtime import (
    PolicySetBatchGenerationRequest,
    evaluate_quality_units_batched,
)
from quality_teacher_runtime import EvaluationUnit
from quality_ranker_protected import ProtectedObservationError, load_observation_universe


PANEL = ROOT / "configs" / "quality_teacher_nemotron_single_v1.json"
LUNA_PANEL = ROOT / "configs" / "quality_teacher_luna_single_v1.json"
POLICY_ID = "q3_substantive_payload"


def _vote(decision: PolicyDecision, reason_code: str) -> TeacherVote:
    return TeacherVote(
        teacher_id="nemotron-3-ultra-build-single-v1",
        policy_id=POLICY_ID,
        decision=decision,
        reason_codes=(reason_code,),
    )


class RepeatedFailAdapter:
    def __init__(self) -> None:
        self.calls: list[PolicySetBatchGenerationRequest] = []

    def generate_policy_batch(self, request: PolicySetBatchGenerationRequest) -> str:
        self.calls.append(request)
        reason_by_policy = {
            "q1_correctness_evidence": "observable_correctness_evidence",
            "q2_semantic_coherence": "recoverable_semantic_unit",
            "q3_substantive_payload": "boilerplate_only",
            "q4_learnable_relations": "recoverable_relation_present",
        }
        return json.dumps(
            {
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "policies": [
                            {
                                "policy_id": policy.policy_id,
                                "decision": (
                                    "fail"
                                    if policy.policy_id == "q3_substantive_payload"
                                    else "pass"
                                ),
                                "reason_codes": [reason_by_policy[policy.policy_id]],
                            }
                            for policy in request.policies
                        ],
                    }
                    for unit in request.units
                ]
            }
        )


def test_single_teacher_panel_is_calibration_only() -> None:
    # Given: the frozen single-Nemotron calibration manifest.
    # When: the manifest is parsed at the configuration boundary.
    panel = load_teacher_panel(PANEL)

    # Then: the teacher cannot directly delete runtime data.
    assert panel.aggregation_strategy == "single_teacher_confirmed_fail"
    assert len(panel.teachers) == 1
    assert panel.teacher_output_alone_may_delete is False
    assert panel.lifecycle == "single_teacher_calibration_oracle"


def test_luna_batch_panel_freezes_openai_provider_identity() -> None:
    panel = load_teacher_panel(LUNA_PANEL)

    assert panel.schema_version == "quality-teacher-panel-v3"
    assert panel.aggregation_strategy == "single_teacher_confirmed_fail"
    assert len(panel.teachers) == 1
    teacher = panel.teachers[0]
    assert teacher.location.value == "openai"
    assert teacher.model_id == "gpt-5.6-luna"
    assert teacher.api_key_environment_variable == "OPENAI_API_KEY"
    assert teacher.reasoning_control == "reasoning_effort_low"


def test_single_teacher_pass_is_accepted_without_repetition() -> None:
    # Given: Nemotron returns a valid pass label.
    first = (_vote(PolicyDecision.PASS, "substantive_payload_present"),)

    # When: the calibration decision is calculated.
    decision = decide_single_teacher(first_pass=first, second_pass=None)

    # Then: the pass label is accepted for ranker calibration.
    assert decision is PanelDecision.PASS


def test_single_teacher_fail_requires_matching_blinded_repetition() -> None:
    # Given: Nemotron repeats the same fail evidence in a blinded pass.
    first = (_vote(PolicyDecision.FAIL, "boilerplate_only"),)
    second = (_vote(PolicyDecision.FAIL, "boilerplate_only"),)

    # When: the calibration decision is calculated.
    decision = decide_single_teacher(first_pass=first, second_pass=second)

    # Then: only the repeated fail becomes a trainable fail label.
    assert decision is PanelDecision.FAIL


def test_single_teacher_changed_fail_abstains() -> None:
    # Given: the repeated run changes either decision or evidence.
    first = (_vote(PolicyDecision.FAIL, "boilerplate_only"),)
    changed_decision = (_vote(PolicyDecision.PASS, "substantive_payload_present"),)
    changed_reason = (_vote(PolicyDecision.FAIL, "no_substantive_residual"),)

    # When/Then: both unstable outcomes fail closed to abstention.
    assert decide_single_teacher(first, changed_decision) is PanelDecision.ABSTAIN
    assert decide_single_teacher(first, changed_reason) is PanelDecision.ABSTAIN


def test_single_teacher_abstain_is_retained_without_repetition() -> None:
    # Given: Nemotron cannot establish the policy boundary.
    first = (_vote(PolicyDecision.ABSTAIN, "specialized_payload_uncertain"),)

    # When: the calibration decision is calculated.
    decision = decide_single_teacher(first_pass=first, second_pass=None)

    # Then: uncertainty remains abstention and cannot become a delete label.
    assert decision is PanelDecision.ABSTAIN


def test_batched_single_teacher_repeats_only_fail_candidates() -> None:
    # Given: one unit whose Q3 policy repeatedly fails while Q1, Q2, and Q4 pass.
    panel = load_teacher_panel(PANEL)
    adapter = RepeatedFailAdapter()
    unit = EvaluationUnit(
        unit_id="single-teacher-batch",
        text="Generated navigation boilerplate.",
        declared_context=None,
        attached_evidence=(),
    )

    # When: the real batched aggregation path evaluates the unit.
    result = evaluate_quality_units_batched(
        panel,
        {panel.teachers[0].teacher_id: adapter},
        (unit,),
    )[0]

    # Then: Q3 requires pass two, while accepted pass labels do not.
    by_policy = {item.policy_id: item for item in result.evidence.policy_results}
    assert by_policy["q3_substantive_payload"].decision is PanelDecision.FAIL
    assert by_policy["q3_substantive_payload"].second_pass is not None
    assert by_policy["q2_semantic_coherence"].decision is PanelDecision.PASS
    assert by_policy["q2_semantic_coherence"].second_pass is None
    assert len(adapter.calls) == 2


def test_ranker_loader_requires_explicit_single_teacher_authority() -> None:
    # Given: one confirmed observation and one otherwise identical legacy-shaped row.
    policy_results = [
        {"policy_id": policy_id, "panel_decision": "pass"}
        for policy_id in (
            "q1_correctness_evidence",
            "q2_semantic_coherence",
            "q3_substantive_payload",
            "q4_learnable_relations",
        )
    ]
    base = {
        "teacher_panel_sha256": "a" * 64,
        "quality_runtime_sha256": "b" * 64,
        "chunk_uid": "single-authority",
        "text_sha256": "c" * 64,
        "available_teacher_ids": ["nemotron-3-ultra-build-single-v1"],
        "policy_results": policy_results,
    }
    with TemporaryDirectory() as directory:
        accepted_path = Path(directory) / "accepted.jsonl"
        accepted_path.write_text(
            json.dumps({**base, "aggregation_strategy": "single_teacher_confirmed_fail"})
            + "\n",
            encoding="utf-8",
        )
        rejected_path = Path(directory) / "rejected.jsonl"
        rejected_path.write_text(json.dumps(base) + "\n", encoding="utf-8")

        # When: the ranker parses the explicitly authorized single-teacher evidence.
        accepted = load_observation_universe((accepted_path,))

        # Then: it is accepted, while an unmarked one-teacher row is rejected.
        assert len(accepted.observations) == 1
        try:
            load_observation_universe((rejected_path,))
        except ProtectedObservationError as error:
            assert error.reason_code == "teacher_observation_universe_missing"
        else:
            raise AssertionError("Unmarked one-teacher evidence must not train the ranker")


if __name__ == "__main__":
    test_single_teacher_panel_is_calibration_only()
    test_luna_batch_panel_freezes_openai_provider_identity()
    test_single_teacher_pass_is_accepted_without_repetition()
    test_single_teacher_fail_requires_matching_blinded_repetition()
    test_single_teacher_changed_fail_abstains()
    test_single_teacher_abstain_is_retained_without_repetition()
    test_batched_single_teacher_repeats_only_fail_candidates()
    test_ranker_loader_requires_explicit_single_teacher_authority()
    print("[quality-teacher-single-runtime-v1] calibration authority: pass")
