#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import load_teacher_panel
from quality_teacher_qualification_runner import (
    IncompatibleObservationError,
    OBSERVATION_SCHEMA_VERSION,
    build_qualification_tasks,
    load_completed_task_ids,
    run_tasks,
)
from quality_teacher_runtime import TeacherGenerationRequest


CONFIG = ROOT / "configs" / "quality_teacher_panel_v1.json"


class PolicyAwareAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, request: TeacherGenerationRequest) -> str:
        self.calls.append(request.unit_id)
        reason = request.pass_reason_codes[0]
        return f'{{"decision":"pass","reason_codes":["{reason}"]}}'


def test_runner_is_resumable_and_emits_no_raw_model_text() -> None:
    panel = load_teacher_panel(CONFIG)
    tasks = tuple(
        task
        for task in build_qualification_tasks("behavior")
        if task.policy_id == "q2_semantic_coherence"
    )[:2]
    adapters = {teacher.teacher_id: PolicyAwareAdapter() for teacher in panel.teachers}

    records = run_tasks(panel, adapters, tasks, completed_task_ids={tasks[0].task_id})

    assert len(records) == 1
    assert records[0]["task_id"] == tasks[1].task_id
    assert records[0]["panel_decision"] == "pass"
    assert "raw_response" not in str(records[0])
    assert all(len(adapter.calls) == 1 for adapter in adapters.values())
    assert records[0]["schema_version"] == OBSERVATION_SCHEMA_VERSION


def test_resume_rejects_observations_from_an_older_runtime_contract() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "observations.jsonl"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "quality-teacher-observation-v1",
                    "task_id": "legacy-task",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            load_completed_task_ids(path)
        except IncompatibleObservationError as error:
            assert error.observed_schema == "quality-teacher-observation-v1"
        else:
            raise AssertionError("Legacy observations must not enter the v2 qualification run")


def test_protected_tasks_apply_all_four_policies() -> None:
    tasks = build_qualification_tasks("protected")

    assert len(tasks) == 800 * 4
    assert len({task.task_id for task in tasks}) == 3200
    assert {task.policy_id for task in tasks} == {
        "q1_correctness_evidence",
        "q2_semantic_coherence",
        "q3_substantive_payload",
        "q4_learnable_relations",
    }
    assert {task.expected_decision for task in tasks} == {"pass"}


if __name__ == "__main__":
    test_runner_is_resumable_and_emits_no_raw_model_text()
    test_resume_rejects_observations_from_an_older_runtime_contract()
    test_protected_tasks_apply_all_four_policies()
    print("[quality-qualification-runner-v1] resumable behavior/protected execution: pass")
