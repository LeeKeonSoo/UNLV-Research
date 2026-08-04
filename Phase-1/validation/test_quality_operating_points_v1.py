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


def _votes(*decisions: PolicyDecision) -> tuple[TeacherVote, ...]:
    return tuple(
        TeacherVote(
            teacher_id=f"teacher-{index}",
            policy_id="q3_substantive_payload",
            decision=decision,
            reason_codes=("controlled_reason",),
        )
        for index, decision in enumerate(decisions)
    )


def _result(
    decision: PanelDecision,
    first: tuple[TeacherVote, ...],
    second: tuple[TeacherVote, ...] | None = None,
) -> PanelPolicyResult:
    return PanelPolicyResult("q3_substantive_payload", decision, first, second)


def test_normal_requires_first_pass_unanimous_fail() -> None:
    unanimous = _result(
        PanelDecision.FAIL,
        _votes(PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.FAIL),
    )
    stable_majority = _result(
        PanelDecision.FAIL,
        _votes(PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.PASS),
        _votes(PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.ABSTAIN),
    )

    assert decide_quality_action((unanimous,), CurationMode.NORMAL, False).action is QualityAction.REMOVE
    assert decide_quality_action((stable_majority,), CurationMode.NORMAL, False).action is QualityAction.RETAIN
    assert decide_quality_action((stable_majority,), CurationMode.HARD, False).action is QualityAction.REMOVE


def test_abstention_and_coverage_veto_retain() -> None:
    failed = _result(
        PanelDecision.FAIL,
        _votes(PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.FAIL),
    )
    abstained = _result(
        PanelDecision.ABSTAIN,
        _votes(PolicyDecision.ABSTAIN, PolicyDecision.PASS, PolicyDecision.FAIL),
    )

    assert decide_quality_action((abstained,), CurationMode.HARD, False).action is QualityAction.RETAIN
    vetoed = decide_quality_action((failed,), CurationMode.HARD, True)
    assert vetoed.action is QualityAction.RETAIN
    assert vetoed.reason_code == "coverage_veto_retain"


if __name__ == "__main__":
    test_normal_requires_first_pass_unanimous_fail()
    test_abstention_and_coverage_veto_retain()
    print("[quality-operating-points-v1] Normal/Hard consensus strength and Coverage veto: pass")
