#!/usr/bin/env python3
"""Normalize frozen official Code benchmark sources for contamination auditing."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else ""


def _jsonl(path: Path) -> Iterable[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def livecodebench_tasks(paths: Iterable[Path]) -> list[JsonMap]:
    tasks: list[JsonMap] = []
    for path in paths:
        for row in _jsonl(path):
            tasks.append(
                {
                    "task_id": str(row["question_id"]),
                    "prompt": "\n".join(part for part in (_text(row.get("starter_code")), _text(row.get("question_content"))) if part),
                    "test": _text(row.get("public_test_cases")),
                }
            )
    return tasks


def bigcodebench_tasks(rows: Iterable[JsonMap]) -> list[JsonMap]:
    return [
        {
            "task_id": str(row["task_id"]),
            "prompt": _text(row.get("complete_prompt")),
            "canonical_solution": _text(row.get("canonical_solution")),
            "test": _text(row.get("test")),
        }
        for row in rows
    ]


def cruxeval_tasks(path: Path) -> list[JsonMap]:
    return [
        {
            "task_id": str(row["id"]),
            "code": _text(row.get("code")),
            "assertion": json.dumps({"input": row.get("input"), "output": row.get("output")}, ensure_ascii=False, sort_keys=True),
        }
        for row in _jsonl(path)
    ]


def ds1000_tasks(path: Path) -> list[JsonMap]:
    tasks: list[JsonMap] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            tasks.append(
                {
                    "task_id": str(metadata.get("problem_id") or f"ds1000/{index}"),
                    "prompt": _text(row.get("prompt")),
                    "canonical_solution": _text(row.get("reference_code")),
                    "text": _text(row.get("code_context")),
                }
            )
    return tasks


def build_snapshot(benchmark_id: str, revision: str, tasks: list[JsonMap]) -> JsonMap:
    return {
        "benchmark_id": benchmark_id,
        "snapshot_revision": revision,
        "tasks": tasks,
        "source_task_count": len(tasks),
    }


def _write_snapshot(path: Path, snapshot: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official Code benchmark contamination snapshots.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--livecodebench-file", action="append", required=True, type=Path)
    parser.add_argument("--livecodebench-revision", required=True)
    parser.add_argument("--bigcodebench-parquet", required=True, type=Path)
    parser.add_argument("--bigcodebench-revision", required=True)
    parser.add_argument("--cruxeval-jsonl", required=True, type=Path)
    parser.add_argument("--cruxeval-revision", required=True)
    parser.add_argument("--ds1000-jsonl-gz", required=True, type=Path)
    parser.add_argument("--ds1000-revision", required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    bigcodebench = load_dataset("parquet", data_files=str(args.bigcodebench_parquet), split="train")
    snapshots = {
        "livecodebench_code_generation_lite": build_snapshot(
            "livecodebench_code_generation_lite", args.livecodebench_revision, livecodebench_tasks(args.livecodebench_file)
        ),
        "bigcodebench_complete": build_snapshot(
            "bigcodebench_complete", args.bigcodebench_revision, bigcodebench_tasks(bigcodebench)
        ),
        "cruxeval_input_prediction": build_snapshot(
            "cruxeval_input_prediction", args.cruxeval_revision, cruxeval_tasks(args.cruxeval_jsonl)
        ),
        "cruxeval_output_prediction": build_snapshot(
            "cruxeval_output_prediction", args.cruxeval_revision, cruxeval_tasks(args.cruxeval_jsonl)
        ),
        "ds1000": build_snapshot("ds1000", args.ds1000_revision, ds1000_tasks(args.ds1000_jsonl_gz)),
    }
    for benchmark_id, snapshot in snapshots.items():
        _write_snapshot(args.output_dir / f"{benchmark_id}.json", snapshot)
    print(json.dumps({benchmark_id: snapshot["source_task_count"] for benchmark_id, snapshot in snapshots.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
