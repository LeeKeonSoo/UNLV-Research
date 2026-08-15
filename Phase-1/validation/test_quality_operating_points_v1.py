#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_operating_points import CurationMode, QualityAction, decide_quality_action
from quality_teacher_panel import PanelDecision, PolicyDecision, TeacherVote
from quality_teacher_runtime import PanelPolicyResult


def _votes(policy_id: str, *decisions: PolicyDecision) -> tuple[TeacherVote, ...]:
    return tuple(
        TeacherVote(
            teacher_id=f"teacher-{index}",
            policy_id=policy_id,
            decision=decision,
            reason_codes=("controlled_reason",),
        )
        for index, decision in enumerate(decisions)
    )


def _result(
    policy_id: str,
    decision: PanelDecision,
    first: tuple[TeacherVote, ...],
    second: tuple[TeacherVote, ...] | None = None,
) -> PanelPolicyResult:
    return PanelPolicyResult(policy_id, decision, first, second)


def test_normal_requires_one_pass_and_hard_requires_two_passes() -> None:
    q2_pass = _result(
        "q2_semantic_coherence",
        PanelDecision.PASS,
        _votes("q2_semantic_coherence", PolicyDecision.PASS, PolicyDecision.PASS, PolicyDecision.PASS),
    )
    q4_pass = _result(
        "q4_learnable_relations",
        PanelDecision.PASS,
        _votes("q4_learnable_relations", PolicyDecision.PASS, PolicyDecision.PASS, PolicyDecision.PASS),
    )

    assert decide_quality_action((q2_pass,), CurationMode.NORMAL, False).action is QualityAction.RETAIN
    assert decide_quality_action((q2_pass,), CurationMode.HARD, False).action is QualityAction.NOT_SELECT
    assert decide_quality_action((q2_pass, q4_pass), CurationMode.HARD, False).action is QualityAction.RETAIN


def test_abstention_is_not_selected_and_coverage_may_restore() -> None:
    failed = _result(
        "q3_substantive_payload",
        PanelDecision.FAIL,
        _votes("q3_substantive_payload", PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.FAIL),
    )
    abstained = _result(
        "q3_substantive_payload",
        PanelDecision.ABSTAIN,
        _votes("q3_substantive_payload", PolicyDecision.ABSTAIN, PolicyDecision.PASS, PolicyDecision.FAIL),
    )

    assert decide_quality_action((abstained,), CurationMode.NORMAL, False).action is QualityAction.NOT_SELECT
    vetoed = decide_quality_action((failed,), CurationMode.HARD, True)
    assert vetoed.action is QualityAction.RETAIN
    assert vetoed.reason_code == "coverage_veto_retain"


def test_declared_verifier_fail_is_strong_in_both_modes() -> None:
    verified = PanelPolicyResult(
        "q1_correctness_evidence",
        PanelDecision.FAIL,
        (),
        None,
        decision_source="declared_verifier",
        reason_codes=("declared_verifier_failed",),
    )

    assert decide_quality_action((verified,), CurationMode.NORMAL, False).action is QualityAction.NOT_SELECT
    assert decide_quality_action((verified,), CurationMode.HARD, False).action is QualityAction.NOT_SELECT


if __name__ == "__main__":
    test_normal_requires_one_pass_and_hard_requires_two_passes()
    test_abstention_is_not_selected_and_coverage_may_restore()
    test_declared_verifier_fail_is_strong_in_both_modes()
    print("[quality-operating-points-v1] Normal/Hard consensus strength and Coverage veto: pass")
