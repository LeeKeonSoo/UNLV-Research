#!/usr/bin/env python3
"""Evaluate DS-1000 generations on Windows using subprocess isolation.

DS-1000's upstream evaluator depends on Unix ``fork`` and ``setitimer``.
This runner preserves the upstream test programs and pass/fail contract while
using one bounded Windows subprocess per problem.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from typing import TypedDict


class ProblemMetadata(TypedDict):
    problem_id: int
    library: str
    perturbation_type: str


class Problem(TypedDict):
    code_context: str
    metadata: ProblemMetadata


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    problem_id: int
    library: str
    perturbation_type: str
    passed: bool
    status: str
    diagnostic: str | None = None


DEFAULT_DATA_CACHE = Path(gettempdir()) / "unlv-ds1000-data-cache"


def _require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def load_problems(path: Path) -> list[Problem]:
    problems: list[Problem] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _require_mapping(json.loads(line), label=f"problem {line_number}")
            metadata = _require_mapping(
                record.get("metadata"), label=f"problem {line_number} metadata"
            )
            problem_id = metadata.get("problem_id")
            library = metadata.get("library")
            perturbation_type = metadata.get("perturbation_type")
            code_context = record.get("code_context")
            if not isinstance(problem_id, int):
                raise TypeError(f"problem {line_number} has no integer problem_id")
            if not isinstance(library, str) or not isinstance(perturbation_type, str):
                raise TypeError(f"problem {line_number} has invalid metadata")
            if not isinstance(code_context, str):
                raise TypeError(f"problem {line_number} has no code_context")
            problems.append(
                {
                    "code_context": code_context,
                    "metadata": {
                        "problem_id": problem_id,
                        "library": library,
                        "perturbation_type": perturbation_type,
                    },
                }
            )
    return problems


def postprocess_answer(value: str | list[str]) -> str:
    code = value[0] if isinstance(value, list) and value else value
    if not isinstance(code, str):
        raise TypeError("DS-1000 answer code must be a string or non-empty string list")
    code = code.split("</code>", 1)[0]
    code = code.replace("```python", "")
    code = code.split("```", 1)[0]
    code = code.split("\nEND SOLUTION", 1)[0]
    return code.replace("<code>", "")


def load_answers(path: Path) -> list[str]:
    answers: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _require_mapping(json.loads(line), label=f"answer {line_number}")
            raw_code = record.get("code")
            if not isinstance(raw_code, (str, list)):
                raise TypeError(f"answer {line_number} has invalid code")
            answers.append(postprocess_answer(raw_code))
    return answers


def load_reference_answers(path: Path) -> list[str]:
    answers: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _require_mapping(json.loads(line), label=f"problem {line_number}")
            reference_code = record.get("reference_code")
            if not isinstance(reference_code, str):
                raise TypeError(
                    f"problem {line_number} has no string reference_code"
                )
            answers.append(postprocess_answer(reference_code))
    return answers


def build_test_program(problem: Problem, answer: str) -> str:
    code_context = problem["code_context"]
    string_test = "test_string(code)\n" if "test_string(" in code_context else ""
    return (
        f"{code_context}\n"
        f"code = {answer!r}\n"
        "test_execution(code)\n"
        f"{string_test}"
    )


def evaluate_problem(
    problem: Problem,
    answer: str,
    *,
    python_executable: Path,
    timeout_seconds: float,
    data_cache: Path = DEFAULT_DATA_CACHE,
) -> EvaluationResult:
    metadata = problem["metadata"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(data_cache / "matplotlib")
    environment["SEABORN_DATA"] = str(data_cache / "seaborn")
    environment["SCIKIT_LEARN_DATA"] = str(data_cache / "scikit_learn")
    with TemporaryDirectory(prefix="ds1000-") as directory:
        script = Path(directory) / "evaluate.py"
        execution_log = Path(directory) / "execution.log"
        script.write_text(build_test_program(problem, answer), encoding="utf-8")
        try:
            with execution_log.open("w", encoding="utf-8", errors="replace") as log:
                completed = subprocess.run(
                    [str(python_executable), str(script)],
                    cwd=directory,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            passed = completed.returncode == 0
            status = "passed" if passed else f"failed_exit_{completed.returncode}"
            diagnostic = None
            if not passed:
                diagnostic = execution_log.read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
        except subprocess.TimeoutExpired:
            passed = False
            status = "timed_out"
            diagnostic = f"Exceeded {timeout_seconds:.3f}s wall-clock timeout"
    return EvaluationResult(
        problem_id=metadata["problem_id"],
        library=metadata["library"],
        perturbation_type=metadata["perturbation_type"],
        passed=passed,
        status=status,
        diagnostic=diagnostic,
    )


def evaluate_corpus(
    problems: list[Problem],
    answers: list[str],
    *,
    python_executable: Path,
    timeout_seconds: float,
    workers: int,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 25,
    data_cache: Path = DEFAULT_DATA_CACHE,
) -> list[EvaluationResult]:
    if len(problems) != len(answers):
        raise ValueError(
            f"DS-1000 problem/answer count mismatch: {len(problems)} != {len(answers)}"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    results: list[EvaluationResult] = []
    if checkpoint_path is not None:
        _write_results(checkpoint_path, results)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                evaluate_problem,
                problem,
                answer,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
                data_cache=data_cache,
            )
            for problem, answer in zip(problems, answers, strict=True)
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if checkpoint_path is not None and (
                completed_count % checkpoint_interval == 0
                or completed_count == len(futures)
            ):
                _write_results(checkpoint_path, results)
            if completed_count % 25 == 0 or completed_count == len(futures):
                print(f"DS-1000 scored {completed_count}/{len(futures)}", flush=True)
    return sorted(results, key=lambda result: result.problem_id)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_results(path: Path, results: list[EvaluationResult]) -> None:
    ordered = sorted(results, key=lambda result: result.problem_id)
    _atomic_write_text(
        path,
        json.dumps([asdict(result) for result in ordered], indent=2),
    )


def render_summary(results: list[EvaluationResult]) -> str:
    if not results:
        raise ValueError("DS-1000 produced no evaluation results")
    mean = sum(result.passed for result in results) / len(results)
    lines = [
        "      score",
        f"count  {len(results):.3f}",
        f"mean      {mean:.6f}",
        "",
        "library count passed mean",
    ]
    for library in sorted({result.library for result in results}):
        group = [result for result in results if result.library == library]
        group_mean = sum(result.passed for result in group) / len(group)
        lines.append(
            f"{library} {len(group)} {sum(result.passed for result in group)} "
            f"{group_mean:.6f}"
        )
    lines.extend(("", "status count"))
    for status, count in sorted(Counter(result.status for result in results).items()):
        lines.append(f"{status} {count}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    answer_source = parser.add_mutually_exclusive_group(required=True)
    answer_source.add_argument("--answers", type=Path)
    answer_source.add_argument(
        "--reference-code",
        action="store_true",
        help="Validate the dataset's frozen reference_code solutions.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--details", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--data-cache", type=Path, default=DEFAULT_DATA_CACHE)
    parser.add_argument(
        "--problem-ids",
        help="Optional comma-separated problem IDs for an environment diagnostic subset.",
    )
    args = parser.parse_args()

    problems = load_problems(args.dataset)
    answers = (
        load_reference_answers(args.dataset)
        if args.reference_code
        else load_answers(args.answers)
    )
    if args.problem_ids:
        requested_ids = {
            int(value.strip())
            for value in args.problem_ids.split(",")
            if value.strip()
        }
        selected = [
            (problem, answer)
            for problem, answer in zip(problems, answers, strict=True)
            if problem["metadata"]["problem_id"] in requested_ids
        ]
        found_ids = {problem["metadata"]["problem_id"] for problem, _ in selected}
        if found_ids != requested_ids:
            raise ValueError(
                f"Unknown DS-1000 problem IDs: {sorted(requested_ids - found_ids)}"
            )
        problems = [problem for problem, _ in selected]
        answers = [answer for _, answer in selected]
    details = args.details or args.result.with_suffix(".json")
    results = evaluate_corpus(
        problems,
        answers,
        python_executable=args.python,
        timeout_seconds=args.timeout_seconds,
        workers=args.workers,
        checkpoint_path=details,
        data_cache=args.data_cache,
    )
    _atomic_write_text(args.result, render_summary(results))
    _write_results(details, results)
    print(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
