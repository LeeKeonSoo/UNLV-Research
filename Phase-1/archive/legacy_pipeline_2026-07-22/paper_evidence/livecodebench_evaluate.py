from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias, TypedDict

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class EvaluationReport(TypedDict):
    schema_version: str
    status: str
    arm: str
    metric: str
    task_count: int
    pass_count: int
    pass_rate: float
    strata: JsonValue
    rows: JsonValue
    tasks_sha256: str
    generations_sha256: str
    official_runner_commit: str
    elapsed_seconds: float
    isolation: str
    utility_stage: str
    selector_tuning_permission: bool


@dataclass(frozen=True, slots=True)
class MissingGenerationError(RuntimeError):
    question_ids: tuple[str, ...]

    def __str__(self) -> str:
        return f"Missing generations: {self.question_ids}"


def summarize_strata(
    rows: Sequence[Mapping[str, str | bool]],
) -> tuple[dict[str, str | int | float], ...]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (str(row["platform"]), str(row["difficulty"]))
        counts[key][0] += 1
        counts[key][1] += int(row["passed"] is True)
    return tuple(
        {
            "platform": platform,
            "difficulty": difficulty,
            "task_count": task_count,
            "pass_count": pass_count,
            "pass_rate": pass_count / task_count,
        }
        for (platform, difficulty), (task_count, pass_count) in sorted(counts.items())
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    *,
    tasks_path: Path,
    generations_path: Path,
    output_path: Path,
    arm: str,
    lcb_repo: Path,
) -> EvaluationReport:
    sys.path.insert(0, str(lcb_repo))
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics

    task_rows = json.loads(tasks_path.read_text(encoding="utf-8"))
    generation_rows = json.loads(generations_path.read_text(encoding="utf-8"))
    generation_map = {
        str(row["question_id"]): [str(code) for code in row["code_list"]]
        for row in generation_rows
    }
    problems = sorted(
        (CodeGenerationProblem(**row) for row in task_rows),
        key=lambda problem: problem.question_id,
    )
    missing = [problem.question_id for problem in problems if problem.question_id not in generation_map]
    if missing:
        raise MissingGenerationError(tuple(missing))
    samples = [problem.get_evaluation_sample() for problem in problems]
    generations = [generation_map[problem.question_id] for problem in problems]
    started = time.time()
    metrics, _, metadata = codegen_metrics(
        samples,
        generations,
        k_list=[1],
        num_process_evaluate=4,
        timeout=6,
        debug=False,
    )
    detail = metrics["detail"]["pass@1"]
    rows = tuple(
        {
            "question_id": problem.question_id,
            "platform": problem.platform.value,
            "difficulty": problem.difficulty.value,
            "passed": bool(detail[index]),
            "metadata": metadata[index],
        }
        for index, problem in enumerate(problems)
    )
    pass_count = sum(row["passed"] is True for row in rows)
    report: EvaluationReport = {
        "schema_version": "code-livecodebench-pilot-evaluation-v1",
        "status": "evaluation_completed",
        "arm": arm,
        "metric": "pass@1",
        "task_count": len(rows),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(rows),
        "strata": summarize_strata(rows),
        "rows": rows,
        "tasks_sha256": _sha256(tasks_path),
        "generations_sha256": _sha256(generations_path),
        "official_runner_commit": (lcb_repo / ".git" / "HEAD").read_text(encoding="ascii").strip(),
        "elapsed_seconds": round(time.time() - started, 3),
        "isolation": "Docker Linux, network disabled",
        "utility_stage": "Stage C only",
        "selector_tuning_permission": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "arm", "task_count", "pass_count", "pass_rate")}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--lcb-repo", type=Path, default=Path("/opt/livecodebench"))
    args = parser.parse_args()
    evaluate(
        tasks_path=args.tasks,
        generations_path=args.generations,
        output_path=args.output,
        arm=args.arm,
        lcb_repo=args.lcb_repo,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
