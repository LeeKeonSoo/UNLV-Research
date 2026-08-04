from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from quality_teacher_qualification import ProtectedFixtureGate
from quality_teacher_qualification_runner import build_qualification_tasks


def _unique_by_task(records: Iterable[dict[str, object]]) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        task_id = str(record["task_id"])
        if task_id in indexed:
            raise ValueError(f"Duplicate qualification task: {task_id}")
        indexed[task_id] = record
    return indexed


def _first_pass_unanimous_fail(record: dict[str, object]) -> bool:
    if record.get("decision_source") == "declared_verifier":
        return record.get("panel_decision") == "fail"
    votes = record.get("first_pass", [])
    return (
        isinstance(votes, list)
        and len(votes) == 3
        and all(isinstance(vote, dict) and vote.get("decision") == "fail" for vote in votes)
    )


def _matrix_summary(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    counts: Counter[tuple[str, str, str, bool]] = Counter()
    for record in records:
        key = (
            str(record["policy_id"]),
            str(record["route"]),
            str(record.get("fixture_class", "unknown")),
            record["expected_decision"] == record["panel_decision"],
        )
        counts[key] += 1
    cells: list[dict[str, object]] = []
    for (policy_id, route, fixture_class, matched), count in sorted(counts.items()):
        cells.append(
            {
                "policy_id": policy_id,
                "route": route,
                "fixture_class": fixture_class,
                "matched": matched,
                "count": count,
            }
        )
    return cells


def build_qualification_report(
    behavior_records: Iterable[dict[str, object]],
    protected_records: Iterable[dict[str, object]],
) -> dict[str, object]:
    behavior = _unique_by_task(behavior_records)
    protected = _unique_by_task(protected_records)
    expected_behavior = {task.task_id for task in build_qualification_tasks("behavior")}
    expected_protected = {task.task_id for task in build_qualification_tasks("protected")}
    behavior_complete = set(behavior) == expected_behavior
    protected_complete = set(protected) == expected_protected
    behavior_matches = sum(
        record["expected_decision"] == record["panel_decision"] for record in behavior.values()
    )

    by_fixture: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in protected.values():
        by_fixture[str(record["fixture_id"])].append(record)
    normal_false = sum(
        any(_first_pass_unanimous_fail(record) for record in records)
        for records in by_fixture.values()
    )
    hard_false = sum(
        any(record["panel_decision"] == "fail" for record in records)
        for records in by_fixture.values()
    )
    fixture_count = len(by_fixture)
    normal_gate = ProtectedFixtureGate(fixture_count, normal_false, 0.005)
    hard_gate = ProtectedFixtureGate(fixture_count, hard_false, 0.02)
    behavior_exact = behavior_complete and behavior_matches == len(expected_behavior)
    report: dict[str, object] = {
        "schema_version": "quality-teacher-qualification-report-v1",
        "behavior_task_count": len(behavior),
        "behavior_expected_task_count": len(expected_behavior),
        "behavior_complete": behavior_complete,
        "behavior_exact_match_count": behavior_matches,
        "behavior_exact": behavior_exact,
        "behavior_matrix": _matrix_summary(behavior.values()),
        "protected_task_count": len(protected),
        "protected_expected_task_count": len(expected_protected),
        "protected_complete": protected_complete,
        "protected_fixture_count": fixture_count,
        "normal_false_removal_count": normal_false,
        "normal_false_removal_upper_bound_95": normal_gate.upper_bound(0.95),
        "hard_false_removal_count": hard_false,
        "hard_false_removal_upper_bound_95": hard_gate.upper_bound(0.95),
        "normal_qualified": behavior_exact and protected_complete and normal_gate.passes(0.95),
        "hard_qualified": behavior_exact and protected_complete and hard_gate.passes(0.95),
        "runtime_activation_mutated": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    canonical = json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Quality teacher qualification report.")
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_qualification_report(_jsonl(args.behavior), _jsonl(args.protected))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"normal_qualified": report["normal_qualified"], "hard_qualified": report["hard_qualified"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
