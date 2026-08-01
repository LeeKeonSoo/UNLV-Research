#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    from pretraining_audit import build_benchmark_exclusion_audit, build_source_snapshot

    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        candidates_path = work_dir / "candidates.jsonl"
        benchmark_path = work_dir / "fixture-benchmark.json"
        audited_path = work_dir / "audited.jsonl"
        _write_jsonl(
            candidates_path,
            [
                {
                    "record_id": "safe",
                    "text": "def stable_sum(values):\n    return sum(values)\n\nThis record documents a deterministic implementation.",
                    "partition": {"source_tier": "raw_like", "source_dataset": "fixture/raw"},
                },
                {
                    "record_id": "overlap",
                    "text": "def benchmark_solution(number):\n    return number * 2\n\nThe task asks for this exact behavior in a benchmark fixture.",
                    "partition": {"reference_pool": "known_high_quality"},
                },
            ],
        )
        benchmark_path.write_text(
            json.dumps(
                {
                    "benchmark_id": "fixturebench",
                    "snapshot_revision": "fixture-revision",
                    "tasks": [
                        {
                            "task_id": "fixture/0",
                            "prompt": "def benchmark_solution(number):\n    return number * 2\n\nThe task asks for this exact behavior in a benchmark fixture.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        snapshot = build_source_snapshot(candidates_path)
        assert snapshot["summary"]["records"] == 2
        assert snapshot["by_source_pool"]["raw_like"]["records"] == 1
        assert snapshot["by_source_pool"]["known_high_quality"]["records"] == 1

        audit = build_benchmark_exclusion_audit(
            candidate_path=candidates_path,
            benchmark_paths=[benchmark_path],
            required_benchmark_ids=["fixturebench", "missingbench"],
            audited_candidate_path=audited_path,
        )
        assert audit["status"] == "benchmark_exclusion_incomplete"
        assert audit["summary"]["excluded_records"] == 1
        assert audit["missing_required_benchmarks"] == ["missingbench"]
        assert [json.loads(line)["record_id"] for line in audited_path.read_text(encoding="utf-8").splitlines()] == ["safe"]

    print("[pretraining-audit] snapshot and incomplete benchmark exclusion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
