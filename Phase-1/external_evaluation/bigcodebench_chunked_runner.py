"""Evaluate BigCodeBench through restart-safe official selective chunks."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Final

from external_evaluation.bigcodebench_remote_runner import (
    DEFAULT_ENDPOINT,
    RemoteEvaluationArtifacts,
    RemoteEvaluationRequest,
    run_remote_evaluation,
)


TASK_ID_PATTERN: Final = re.compile(r"^BigCodeBench/(\d+)$")


@dataclass(frozen=True, slots=True)
class TaskChunk:
    index: int
    task_ids: tuple[str, ...]
    selective_evaluate: str


@dataclass(frozen=True, slots=True)
class ChunkArtifact:
    chunk_index: int
    result_path: Path
    pass_rate_path: Path


@dataclass(frozen=True, slots=True)
class AggregateEvaluation:
    task_count: int
    passed_tasks: int
    pass_at_1: float
    evaluations: Mapping[str, list[Mapping[str, object]]]
    failed_ground_truth_tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChunkedEvaluationRequest:
    samples: Path
    work_root: Path
    final_result_path: Path
    final_eval_path: Path
    chunk_size: int = 50
    expected_task_count: int = 1_140
    parallel: int = 8
    endpoint: str = DEFAULT_ENDPOINT
    max_attempts: int = 5
    retry_seconds: float = 60.0


def _json_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task_ids(path: Path) -> tuple[str, ...]:
    task_ids: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("task_id"), str):
                raise TypeError(f"Invalid task_id at {path}:{line_number}")
            task_ids.append(value["task_id"])
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate task IDs in {path}")
    return tuple(task_ids)


def partition_task_ids(task_ids: Sequence[str], chunk_size: int) -> tuple[TaskChunk, ...]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    numeric_ids: list[int] = []
    for task_id in task_ids:
        match = TASK_ID_PATTERN.fullmatch(task_id)
        if match is None:
            raise ValueError(f"Invalid BigCodeBench task ID: {task_id}")
        numeric_ids.append(int(match.group(1)))
    if numeric_ids != list(range(len(task_ids))):
        raise ValueError("BigCodeBench task IDs must be ordered and contiguous from zero")
    chunks: list[TaskChunk] = []
    for index, start in enumerate(range(0, len(task_ids), chunk_size)):
        selected = tuple(task_ids[start : start + chunk_size])
        selector = ",".join(str(task_id) for task_id in range(start, start + len(selected)))
        chunks.append(TaskChunk(index, selected, selector))
    return tuple(chunks)


def aggregate_chunk_artifacts(
    expected_task_ids: Sequence[str],
    artifacts: Sequence[ChunkArtifact],
) -> AggregateEvaluation:
    evaluations: dict[str, list[Mapping[str, object]]] = {}
    failed_ground_truth: set[str] = set()
    for artifact in artifacts:
        raw_eval = _json_mapping(artifact.result_path).get("eval")
        if not isinstance(raw_eval, dict):
            raise TypeError(f"Missing eval mapping: {artifact.result_path}")
        for task_id, task_results in raw_eval.items():
            if task_id in evaluations:
                raise ValueError(f"Duplicate task judgment: {task_id}")
            if not isinstance(task_results, list) or len(task_results) != 1:
                raise ValueError(f"Expected one judgment for {task_id}")
            result = task_results[0]
            if not isinstance(result, dict) or not isinstance(result.get("status"), str):
                raise TypeError(f"Invalid task judgment for {task_id}")
            evaluations[task_id] = task_results
        raw_failed = _json_mapping(artifact.pass_rate_path).get("failed_tasks", [])
        if not isinstance(raw_failed, list) or not all(
            isinstance(task_id, str) for task_id in raw_failed
        ):
            raise TypeError(f"Invalid failed_tasks: {artifact.pass_rate_path}")
        failed_ground_truth.update(raw_failed)
    expected = set(expected_task_ids)
    actual = set(evaluations)
    if actual != expected:
        raise ValueError(
            f"Incomplete task judgments: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    passed = sum(results[0]["status"] == "pass" for results in evaluations.values())
    return AggregateEvaluation(
        task_count=len(expected_task_ids),
        passed_tasks=passed,
        pass_at_1=passed / len(expected_task_ids),
        evaluations=evaluations,
        failed_ground_truth_tasks=tuple(sorted(failed_ground_truth)),
    )


def run_chunked_evaluation(
    request: ChunkedEvaluationRequest,
    *,
    evaluator: Callable[[RemoteEvaluationRequest], RemoteEvaluationArtifacts] = run_remote_evaluation,
) -> AggregateEvaluation:
    task_ids = load_task_ids(request.samples)
    if len(task_ids) != request.expected_task_count:
        raise ValueError(
            f"Expected {request.expected_task_count} tasks, found {len(task_ids)}"
        )
    chunks = partition_task_ids(task_ids, request.chunk_size)
    artifacts: list[ChunkArtifact] = []
    for chunk in chunks:
        result_path = request.work_root / f"chunk_{chunk.index:03d}_eval_results.json"
        pass_rate_path = request.work_root / f"chunk_{chunk.index:03d}_pass_at_k.json"
        evaluator(
            RemoteEvaluationRequest(
                samples=request.samples,
                result_path=result_path,
                pass_rate_path=pass_rate_path,
                selective_evaluate=chunk.selective_evaluate,
                parallel=request.parallel,
                endpoint=request.endpoint,
                max_attempts=request.max_attempts,
                retry_seconds=request.retry_seconds,
            )
        )
        artifacts.append(ChunkArtifact(chunk.index, result_path, pass_rate_path))
    aggregate = aggregate_chunk_artifacts(task_ids, artifacts)
    source_hash = _sha256(request.samples)
    provenance: dict[str, object] = {
        "schema_version": "bigcodebench-official-chunked-v1",
        "evaluator": "bigcode/bigcodebench-evaluator",
        "aggregation": "exact task-level union; pass count divided by full task count",
        "task_count": aggregate.task_count,
        "passed_tasks": aggregate.passed_tasks,
        "chunk_size": request.chunk_size,
        "chunk_count": len(chunks),
        "source_sha256": source_hash,
        "chunks": [
            {
                "index": chunk.index,
                "selector": chunk.selective_evaluate,
                "task_count": len(chunk.task_ids),
                "result_sha256": _sha256(artifacts[chunk.index].result_path),
                "pass_rate_sha256": _sha256(artifacts[chunk.index].pass_rate_path),
            }
            for chunk in chunks
        ],
    }
    _atomic_json(
        request.final_eval_path,
        {"eval": dict(aggregate.evaluations), "provenance": provenance},
    )
    failed = list(aggregate.failed_ground_truth_tasks)
    _atomic_json(
        request.final_result_path,
        {
            "pass@1": aggregate.pass_at_1,
            "model": request.samples.name,
            "split": "complete",
            "subset": "full",
            "calibrated": True,
            "gt_pass_rate": (aggregate.task_count - len(failed)) / aggregate.task_count,
            "failed_tasks": failed,
            "provenance": provenance,
        },
    )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--final-result", type=Path, required=True)
    parser.add_argument("--final-eval", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    args = parser.parse_args()
    result = run_chunked_evaluation(
        ChunkedEvaluationRequest(
            samples=args.samples,
            work_root=args.work_root,
            final_result_path=args.final_result,
            final_eval_path=args.final_eval,
            chunk_size=args.chunk_size,
            parallel=args.parallel,
            endpoint=args.endpoint,
            max_attempts=args.max_attempts,
            retry_seconds=args.retry_seconds,
        )
    )
    print(
        json.dumps(
            {
                "task_count": result.task_count,
                "passed_tasks": result.passed_tasks,
                "pass@1": result.pass_at_1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
