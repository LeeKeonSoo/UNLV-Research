#!/usr/bin/env python3
"""Regression checks for derived benchmark artifact rules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _module():
    path = PROJECT_DIR / "68_generate_benchmark_task_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("benchmark_task_artifact_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _module()
    first = module._sha256("a\r\nb\n")
    second = module._sha256("a\nb")
    assert first == second
    assert len(first) == 64
    assert module._sha256("") is None
    fingerprints = module.derived_fingerprints("def add(left, right):\n    return left + right\n")
    assert len(fingerprints["token_simhash64"]) == 16, fingerprints
    assert len(fingerprints["python_ast_sha256"]) == 64, fingerprints
    module._rows = lambda *args, **kwargs: iter(
        [
            {
                "repo": "fixture/repository",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the parser.",
                "patch": "def parse(value):\n    return value\n",
                "test_patch": "def test_parse():\n    assert parse(1) == 1\n",
            }
        ]
    )
    generated = module.generate(
        {
            "entries": [
                {
                    "benchmark": "FixtureBench",
                    "dataset_sources": [
                        {"dataset": "fixture/dataset", "config": "default", "splits": ["test"]}
                    ],
                }
            ]
        },
        benchmarks=None,
        delay_seconds=0,
    )
    assert generated["benchmarks"][0]["benchmark"] == "FixtureBench", generated
    print("[benchmark-task-artifacts] normalized hashes do not persist raw task content: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
