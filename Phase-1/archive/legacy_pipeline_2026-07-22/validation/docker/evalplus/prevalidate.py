#!/usr/bin/env python3
"""Run frozen EvalPlus semantic controls inside an isolated Linux container."""

from __future__ import annotations

import json

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.eval import PASS, untrusted_check
from evalplus.gen.util import trusted_exec


SMOKE_TASK_COUNT = 3


def evaluate_task(dataset: str, task):
    reference = str(task["prompt"]) + str(task["canonical_solution"])
    negative = f"def {task['entry_point']}(*args, **kwargs):\n    return None\n"
    controls = {}
    for test_name in ("base", "plus"):
        inputs = task[f"{test_name}_input"]
        expected, ref_time = trusted_exec(reference, inputs, task["entry_point"], record_time=True)
        reference_status, _ = untrusted_check(
            dataset,
            reference,
            inputs,
            task["entry_point"],
            expected=expected,
            atol=task["atol"],
            ref_time=ref_time,
            fast_check=True,
        )
        negative_status, _ = untrusted_check(
            dataset,
            negative,
            inputs,
            task["entry_point"],
            expected=expected,
            atol=task["atol"],
            ref_time=ref_time,
            fast_check=True,
        )
        controls[test_name] = {
            "reference_pass": reference_status == PASS,
            "negative_rejected": negative_status != PASS,
        }
    return {
        "task_id": str(task["task_id"]),
        "reference_all_pass": all(row["reference_pass"] for row in controls.values()),
        "negative_rejected_by_any_suite": any(row["negative_rejected"] for row in controls.values()),
    }


def main():
    datasets = {
        "HumanEval+": ("humaneval", get_human_eval_plus()),
        "MBPP+": ("mbpp", get_mbpp_plus()),
    }
    reports = {}
    for name, (evaluator_name, tasks) in datasets.items():
        selected = sorted(tasks)[:SMOKE_TASK_COUNT]
        rows = [evaluate_task(evaluator_name, tasks[task_id]) for task_id in selected]
        reports[name] = {
            "available_task_count": len(tasks),
            "smoke_task_count": len(rows),
            "reference_control_pass_count": sum(row["reference_all_pass"] for row in rows),
            "negative_control_rejected_count": sum(row["negative_rejected_by_any_suite"] for row in rows),
            "task_ids": selected,
        }
    print(json.dumps({"datasets": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
