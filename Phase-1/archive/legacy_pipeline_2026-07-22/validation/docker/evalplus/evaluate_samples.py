#!/usr/bin/env python3
"""Evaluate a JSONL of EvalPlus samples inside the isolated Docker image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import PASS, untrusted_check
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.gen.util import trusted_exec


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                yield json.loads(value)


def _load_problems(dataset: str) -> Dict[str, Dict[str, Any]]:
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def _solution(problem: Dict[str, Any], sample: Dict[str, Any]) -> str:
    if "solution" in sample:
        return str(sample["solution"])
    return str(problem["prompt"]) + str(sample.get("completion") or "")


def _expected(problem: Dict[str, Any], kind: str) -> Dict[str, Any]:
    output_not_none = problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    expected, ref_time = trusted_exec(
        str(problem["prompt"]) + str(problem["canonical_solution"]),
        problem[f"{kind}_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    return {"expected": expected, "ref_time": ref_time}


def _check(dataset: str, problem: Dict[str, Any], solution: str, kind: str) -> Dict[str, Any]:
    oracle = _expected(problem, kind)
    status, details = untrusted_check(
        dataset,
        solution,
        problem[f"{kind}_input"],
        problem["entry_point"],
        expected=oracle["expected"],
        atol=problem["atol"],
        ref_time=oracle["ref_time"],
        fast_check=True,
    )
    return {
        "status": status,
        "passed": status == PASS,
        "detail_count": len(details or []),
    }


def evaluate(dataset: str, samples_path: Path, output_path: Path) -> Dict[str, Any]:
    problems = _load_problems(dataset)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    for sample in _iter_jsonl(samples_path):
        task_id = str(sample["task_id"])
        if task_id not in problems:
            rows.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": "task_not_in_dataset",
                }
            )
            continue
        problem = problems[task_id]
        try:
            code = _solution(problem, sample)
            base = _check(dataset, problem, code, "base")
            plus = _check(dataset, problem, code, "plus")
            rows.append(
                {
                    "task_id": task_id,
                    "base": base,
                    "plus": plus,
                    "passed": bool(base["passed"] and plus["passed"]),
                }
            )
        except BaseException as exc:  # Eval sandbox exceptions must become evidence rows.
            rows.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    task_count = len(rows)
    pass_count = sum(1 for row in rows if row.get("passed") is True)
    report = {
        "schema_version": "code-domain-evalplus-sample-eval-v1",
        "status": "evalplus_samples_evaluated",
        "dataset": dataset,
        "samples": str(samples_path),
        "task_count": task_count,
        "pass_count": pass_count,
        "pass_rate": pass_count / max(1, task_count),
        "elapsed_seconds": round(time.time() - started, 3),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps({key: report[key] for key in ("status", "dataset", "task_count", "pass_count", "pass_rate")}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EvalPlus samples in Docker.")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.samples, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
