from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.bigcodebench_chunked_runner import (
    ChunkArtifact,
    aggregate_chunk_artifacts,
    partition_task_ids,
)


def _write_chunk(
    root: Path,
    chunk_index: int,
    statuses: dict[str, str],
) -> ChunkArtifact:
    result_path = root / f"chunk_{chunk_index:03d}_eval_results.json"
    pass_rate_path = root / f"chunk_{chunk_index:03d}_pass_at_k.json"
    result_path.write_text(
        json.dumps(
            {
                "eval": {
                    task_id: [{"task_id": task_id, "status": status}]
                    for task_id, status in statuses.items()
                }
            }
        ),
        encoding="utf-8",
    )
    failed_tasks = [
        task_id for task_id, status in statuses.items() if status == "gt_fail"
    ]
    pass_rate_path.write_text(
        json.dumps({"failed_tasks": failed_tasks}),
        encoding="utf-8",
    )
    return ChunkArtifact(
        chunk_index=chunk_index,
        result_path=result_path,
        pass_rate_path=pass_rate_path,
    )


def test_partition_task_ids_is_complete_disjoint_and_deterministic() -> None:
    task_ids = tuple(f"BigCodeBench/{index}" for index in range(7))

    chunks = partition_task_ids(task_ids, chunk_size=3)

    assert [chunk.selective_evaluate for chunk in chunks] == [
        "0,1,2",
        "3,4,5",
        "6",
    ]
    flattened = tuple(task_id for chunk in chunks for task_id in chunk.task_ids)
    assert flattened == task_ids
    assert len(flattened) == len(set(flattened))


def test_aggregate_chunk_artifacts_recomputes_exact_pass_at_one() -> None:
    expected = tuple(f"BigCodeBench/{index}" for index in range(4))
    with TemporaryDirectory() as directory:
        root = Path(directory)
        artifacts = (
            _write_chunk(root, 0, {expected[0]: "pass", expected[1]: "fail"}),
            _write_chunk(root, 1, {expected[2]: "timeout", expected[3]: "pass"}),
        )

        aggregate = aggregate_chunk_artifacts(expected, artifacts)

    assert aggregate.task_count == 4
    assert aggregate.passed_tasks == 2
    assert aggregate.pass_at_1 == 0.5
    assert set(aggregate.evaluations) == set(expected)


def test_aggregate_chunk_artifacts_rejects_missing_or_duplicate_tasks() -> None:
    expected = ("BigCodeBench/0", "BigCodeBench/1")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        duplicate = (
            _write_chunk(root, 0, {"BigCodeBench/0": "pass"}),
            _write_chunk(root, 1, {"BigCodeBench/0": "fail"}),
        )

        try:
            aggregate_chunk_artifacts(expected, duplicate)
        except ValueError as error:
            assert "duplicate" in str(error).lower()
        else:
            raise AssertionError("duplicate task judgments must be rejected")


if __name__ == "__main__":
    test_partition_task_ids_is_complete_disjoint_and_deterministic()
    test_aggregate_chunk_artifacts_recomputes_exact_pass_at_one()
    test_aggregate_chunk_artifacts_rejects_missing_or_duplicate_tasks()
    print("BigCodeBench chunked runner contract passed")
