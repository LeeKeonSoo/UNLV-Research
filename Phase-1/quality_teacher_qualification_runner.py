from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping

from dotenv import load_dotenv

from quality_teacher_fixtures import (
    POLICY_IDS,
    build_behavior_fixture_matrix,
    build_protected_fixture_set,
)
from quality_teacher_panel import QualityPolicy, TeacherPanel, load_teacher_panel
from quality_teacher_runtime import (
    EvaluationUnit,
    PanelPolicyResult,
    TeacherAdapter,
    evaluate_panel_policy,
)


OBSERVATION_SCHEMA_VERSION: Final = "quality-teacher-observation-v2"


@dataclass(frozen=True, slots=True)
class IncompatibleObservationError(RuntimeError):
    path: Path
    observed_schema: str

    def __str__(self) -> str:
        return (
            f"Qualification output {self.path} uses {self.observed_schema!r}; "
            f"expected {OBSERVATION_SCHEMA_VERSION!r}"
        )


@dataclass(frozen=True, slots=True)
class QualificationTask:
    task_id: str
    fixture_id: str
    policy_id: str
    route: str
    expected_decision: Literal["pass", "fail", "abstain"]
    fixture_class: str
    unit: EvaluationUnit


def build_qualification_tasks(
    kind: Literal["behavior", "protected"],
    *,
    samples_per_cell: int | None = None,
) -> tuple[QualificationTask, ...]:
    if kind == "behavior":
        fixture_count = 8 if samples_per_cell is None else samples_per_cell
        if fixture_count < 1:
            raise ValueError("samples_per_cell must be positive")
        return tuple(
            QualificationTask(
                task_id=fixture.fixture_id,
                fixture_id=fixture.fixture_id,
                policy_id=fixture.policy_id,
                route=fixture.route,
                expected_decision=fixture.expected_decision,
                fixture_class=fixture.fixture_class.value,
                unit=fixture.unit,
            )
            for fixture in build_behavior_fixture_matrix(fixture_count)
        )
    if samples_per_cell is not None:
        raise ValueError("samples_per_cell is only valid for behavior qualification")
    return tuple(
        QualificationTask(
            task_id=f"{fixture.fixture_id}-{policy_id}",
            fixture_id=fixture.fixture_id,
            policy_id=policy_id,
            route=fixture.route,
            expected_decision="pass",
            fixture_class="protected_pass",
            unit=fixture.unit,
        )
        for fixture in build_protected_fixture_set()
        for policy_id in POLICY_IDS
    )


def _policy(panel: TeacherPanel, policy_id: str) -> QualityPolicy:
    match = tuple(policy for policy in panel.policies if policy.policy_id == policy_id)
    if len(match) != 1:
        raise ValueError(f"Policy must exist exactly once: {policy_id}")
    return match[0]


def _votes(result: PanelPolicyResult, pass_name: str) -> list[dict[str, object]]:
    votes = result.first_pass if pass_name == "first" else result.second_pass or ()
    return [
        {
            "teacher_id": vote.teacher_id,
            "decision": vote.decision.value,
            "reason_codes": list(vote.reason_codes),
        }
        for vote in votes
    ]


def _task_traces(
    adapters: Mapping[str, TeacherAdapter],
    *,
    unit_id: str,
    policy_id: str,
) -> list[dict[str, object]]:
    traces: list[dict[str, object]] = []
    for adapter in adapters.values():
        available = getattr(adapter, "traces", ())
        traces.extend(
            dict(trace)
            for trace in available
            if trace.get("unit_id") == unit_id and trace.get("policy_id") == policy_id
        )
    return traces


def run_tasks(
    panel: TeacherPanel,
    adapters: Mapping[str, TeacherAdapter],
    tasks: tuple[QualificationTask, ...],
    *,
    completed_task_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    completed = completed_task_ids or set()
    records: list[dict[str, object]] = []
    for task in tasks:
        if task.task_id in completed:
            continue
        result = evaluate_panel_policy(panel, adapters, _policy(panel, task.policy_id), task.unit)
        records.append(
            {
                "schema_version": OBSERVATION_SCHEMA_VERSION,
                "task_id": task.task_id,
                "fixture_id": task.fixture_id,
                "policy_id": task.policy_id,
                "route": task.route,
                "fixture_class": task.fixture_class,
                "expected_decision": task.expected_decision,
                "panel_decision": result.decision.value,
                "decision_source": result.decision_source,
                "decision_reason_codes": list(result.reason_codes),
                "first_pass": _votes(result, "first"),
                "second_pass": _votes(result, "second"),
                "generation_traces": _task_traces(
                    adapters,
                    unit_id=task.unit.unit_id,
                    policy_id=task.policy_id,
                ),
            }
        )
    return records


def load_completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        observed_schema = str(record.get("schema_version", "missing"))
        if observed_schema != OBSERVATION_SCHEMA_VERSION:
            raise IncompatibleObservationError(path=path, observed_schema=observed_schema)
        completed.add(str(record["task_id"]))
    return completed


def append_task_records(
    path: Path,
    panel: TeacherPanel,
    adapters: Mapping[str, TeacherAdapter],
    tasks: tuple[QualificationTask, ...],
    *,
    completed_task_ids: set[str],
    task_workers: int = 1,
) -> int:
    """Appends and flushes each task so a later interruption remains resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_records = 0
    pending = tuple(task for task in tasks if task.task_id not in completed_task_ids)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if task_workers == 1:
            task_groups = ((task, run_tasks(panel, adapters, (task,))) for task in pending)
        else:
            executor = ThreadPoolExecutor(max_workers=task_workers)
            futures = {
                executor.submit(run_tasks, panel, adapters, (task,)): task
                for task in pending
            }
            task_groups = (
                (futures[future], future.result()) for future in as_completed(futures)
            )
        try:
            for task, records in task_groups:
                if not records:
                    continue
                handle.write(json.dumps(records[0], ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                completed_task_ids.add(task.task_id)
                new_records += 1
        finally:
            if task_workers != 1:
                executor.shutdown(wait=True)
    return new_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resumable Quality teacher qualification tasks.")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--kind", choices=("behavior", "protected"), required=True)
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-cell", type=int)
    parser.add_argument("--task-workers", type=int, default=1)
    args = parser.parse_args()

    if not load_dotenv(args.dotenv):
        raise RuntimeError(f"Could not load dotenv: {args.dotenv}")
    panel = load_teacher_panel(args.panel)
    from quality_teacher_smoke import _build_adapters

    if args.task_workers < 1:
        raise ValueError("task_workers must be positive")
    tasks = build_qualification_tasks(
        args.kind,
        samples_per_cell=args.samples_per_cell,
    )[args.offset :]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    adapters = _build_adapters(panel, args.local_model_path)
    new_records = append_task_records(
        args.output,
        panel,
        adapters,
        tasks,
        completed_task_ids=load_completed_task_ids(args.output),
        task_workers=args.task_workers,
    )
    print(json.dumps({"kind": args.kind, "new_records": new_records}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
