#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_qualification_report import build_qualification_report
from quality_teacher_qualification_runner import build_qualification_tasks


def _record(task: object, decision: str | None = None) -> dict[str, object]:
    observed = decision or task.expected_decision
    votes = [
        {"teacher_id": f"teacher-{index}", "decision": observed, "reason_codes": ["fixture"]}
        for index in range(3)
    ]
    return {
        "task_id": task.task_id,
        "fixture_id": task.fixture_id,
        "policy_id": task.policy_id,
        "route": task.route,
        "expected_decision": task.expected_decision,
        "panel_decision": observed,
        "first_pass": votes,
        "second_pass": [],
    }


def test_complete_exact_fixture_evidence_promotes_both_operating_points() -> None:
    behavior = [_record(task) for task in build_qualification_tasks("behavior")]
    protected = [_record(task) for task in build_qualification_tasks("protected")]

    report = build_qualification_report(behavior, protected)

    assert report["behavior_task_count"] == 512
    assert report["behavior_exact_match_count"] == 512
    assert report["protected_fixture_count"] == 800
    assert report["normal_false_removal_count"] == 0
    assert report["hard_false_removal_count"] == 0
    assert report["normal_qualified"] is True
    assert report["hard_qualified"] is True


def test_one_protected_unanimous_fail_blocks_both_modes() -> None:
    behavior = [_record(task) for task in build_qualification_tasks("behavior")]
    tasks = build_qualification_tasks("protected")
    protected = [_record(task, "fail" if index == 0 else None) for index, task in enumerate(tasks)]

    report = build_qualification_report(behavior, protected)

    assert report["normal_false_removal_count"] == 1
    assert report["hard_false_removal_count"] == 1
    assert report["normal_qualified"] is False
    assert report["hard_qualified"] is True


if __name__ == "__main__":
    test_complete_exact_fixture_evidence_promotes_both_operating_points()
    test_one_protected_unanimous_fail_blocks_both_modes()
    print("[quality-qualification-report-v1] completeness and exact confidence gates: pass")
