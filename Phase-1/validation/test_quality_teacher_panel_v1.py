#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import (
    PanelContractError,
    PanelDecision,
    PolicyDecision,
    TeacherVote,
    decide_panel,
    load_teacher_panel,
)


CONFIG = ROOT / "configs" / "quality_teacher_panel_v1.json"


def _vote(teacher_id: str, decision: PolicyDecision) -> TeacherVote:
    return TeacherVote(
        teacher_id=teacher_id,
        policy_id="q1_correctness_evidence",
        decision=decision,
        reason_codes=("controlled_fixture_evidence",),
    )


def test_panel_contract_freezes_diverse_two_hosted_one_local_teachers() -> None:
    # Given: the frozen candidate teacher-panel manifest.
    # When: the manifest is parsed at the configuration boundary.
    panel = load_teacher_panel(CONFIG)

    # Then: the panel is diverse, non-runtime, and has the intended topology.
    assert [teacher.model_id for teacher in panel.teachers] == [
        "google/gemma-4-31b-it",
        "meta/llama-3.1-8b-instruct",
        "Qwen/Qwen3.5-9B",
    ]
    assert len({teacher.organization for teacher in panel.teachers}) == 3
    assert sum(teacher.location.value == "nvidia_build" for teacher in panel.teachers) == 2
    assert sum(teacher.location.value == "local" for teacher in panel.teachers) == 1
    assert panel.runtime_activation is False
    assert panel.teacher_output_alone_may_delete is False
    assert panel.response_contract.maximum_schema_retries == 1
    assert panel.response_contract.invalid_response_action == "abstain"
    assert panel.response_contract.reason_code_pattern == "^[a-z][a-z0-9_]{0,63}$"
    assert all(teacher.maximum_new_tokens == 96 for teacher in panel.teachers)
    hosted = tuple(
        teacher for teacher in panel.teachers if teacher.location.value == "nvidia_build"
    )
    assert all(teacher.request_timeout_seconds == 30 for teacher in hosted)
    assert all(teacher.maximum_transport_retries == 0 for teacher in hosted)
    assert all(teacher.structured_output_mode == "json_object" for teacher in hosted)
    assert panel.policies[0].reason_codes.fail == (
        "reproducible_contradiction",
        "impossible_derivation",
        "declared_verifier_failed",
        "locally_checkable_incorrect_result",
    )
    assert all(policy.reason_codes.pass_ for policy in panel.policies)
    assert all(policy.reason_codes.abstain for policy in panel.policies)


def test_first_pass_unanimity_produces_panel_decision() -> None:
    # Given: three independent teachers return the same first-pass decision.
    votes = tuple(_vote(teacher_id, PolicyDecision.FAIL) for teacher_id in ("a", "b", "c"))

    # When: the panel decision is calculated.
    decision = decide_panel(first_pass=votes, second_pass=None)

    # Then: unanimity is accepted without a second pass.
    assert decision is PanelDecision.FAIL


def test_stable_two_of_three_requires_blinded_second_pass() -> None:
    # Given: the same two teachers agree on both blinded passes.
    first = (
        _vote("a", PolicyDecision.PASS),
        _vote("b", PolicyDecision.PASS),
        _vote("c", PolicyDecision.FAIL),
    )
    second = (
        _vote("a", PolicyDecision.PASS),
        _vote("b", PolicyDecision.PASS),
        _vote("c", PolicyDecision.ABSTAIN),
    )

    # When: both passes are evaluated.
    decision = decide_panel(first_pass=first, second_pass=second)

    # Then: the stable repeated majority is accepted.
    assert decision is PanelDecision.PASS


def test_changed_or_unresolved_second_pass_abstains() -> None:
    # Given: the first-pass majority is not stable under blinded repetition.
    first = (
        _vote("a", PolicyDecision.PASS),
        _vote("b", PolicyDecision.PASS),
        _vote("c", PolicyDecision.FAIL),
    )
    second = (
        _vote("a", PolicyDecision.FAIL),
        _vote("b", PolicyDecision.PASS),
        _vote("c", PolicyDecision.FAIL),
    )

    # When: both passes are evaluated.
    decision = decide_panel(first_pass=first, second_pass=second)

    # Then: disagreement fails closed to abstention.
    assert decision is PanelDecision.ABSTAIN


def test_vote_set_rejects_missing_or_duplicate_teachers() -> None:
    # Given: a malformed vote set repeats one teacher and omits another.
    votes = (
        _vote("a", PolicyDecision.PASS),
        _vote("a", PolicyDecision.PASS),
        _vote("c", PolicyDecision.PASS),
    )

    # When/Then: the boundary rejects the malformed panel input.
    try:
        decide_panel(first_pass=votes, second_pass=None)
    except PanelContractError as error:
        assert "unique teachers" in str(error)
    else:
        raise AssertionError("Duplicate teacher votes must be rejected")


if __name__ == "__main__":
    test_panel_contract_freezes_diverse_two_hosted_one_local_teachers()
    test_first_pass_unanimity_produces_panel_decision()
    test_stable_two_of_three_requires_blinded_second_pass()
    test_changed_or_unresolved_second_pass_abstains()
    test_vote_set_rejects_missing_or_duplicate_teachers()
    print("[quality-teacher-panel-v1] contract and consensus: pass")
