#!/usr/bin/env python3
from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.ds1000_windows_runner import (
    evaluate_corpus,
    load_answers,
    load_problems,
    load_reference_answers,
    render_summary,
)


def test_windows_runner_preserves_ds1000_test_program_contract() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "ds1000.jsonl.gz"
        answers = root / "answers.jsonl"
        checkpoint = root / "details.json"
        problem = {
            "metadata": {
                "problem_id": 7,
                "library": "stdlib",
                "perturbation_type": "fixture",
            },
            "code_context": (
                "def test_execution(code):\n"
                "    namespace = {}\n"
                "    exec(code, namespace)\n"
                "    assert namespace['answer'] == 42\n"
            ),
            "reference_code": "answer = 42",
        }
        with gzip.open(dataset, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(problem) + "\n")
        answers.write_text(json.dumps({"code": "answer = 42"}) + "\n")

        results = evaluate_corpus(
            load_problems(dataset),
            load_answers(answers),
            python_executable=Path(sys.executable),
            timeout_seconds=5.0,
            workers=1,
            checkpoint_path=checkpoint,
            checkpoint_interval=1,
        )

        assert len(results) == 1
        assert results[0].passed is True
        assert "mean      1.000000" in render_summary(results)
        assert load_reference_answers(dataset) == ["answer = 42"]
        assert json.loads(checkpoint.read_text(encoding="utf-8"))[0]["passed"] is True

        failed = evaluate_corpus(
            load_problems(dataset),
            ["answer = 0"],
            python_executable=Path(sys.executable),
            timeout_seconds=5.0,
            workers=1,
        )
        assert failed[0].passed is False
        assert "AssertionError" in (failed[0].diagnostic or "")


if __name__ == "__main__":
    test_windows_runner_preserves_ds1000_test_program_contract()
    print("DS-1000 Windows runner contract passed")
