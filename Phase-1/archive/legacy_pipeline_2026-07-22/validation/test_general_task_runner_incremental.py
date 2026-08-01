#!/usr/bin/env python3
"""Validate incremental general-task runner helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("148_run_code_domain_general_task_guardrail.py")
    base = {
        "results": {"hellaswag": {"acc_norm,none": 0.4}},
        "configs": {"hellaswag": {"task": "hellaswag"}},
    }
    update = {
        "results": {"arc_challenge": {"acc_norm,none": 0.3}},
        "configs": {"arc_challenge": {"task": "arc_challenge"}},
    }
    merged = module._merge_lm_eval_results(base, update)
    assert set(merged["results"]) == {"hellaswag", "arc_challenge"}
    assert set(merged["configs"]) == {"hellaswag", "arc_challenge"}
    assert module._completed_tasks(merged, ["hellaswag", "arc_challenge", "piqa"]) == [
        "hellaswag",
        "arc_challenge",
    ]
    partial = module._build_result_report(
        {"utility_scope": "Stage C validation only; never selector objective"},
        "raw_random_equal_budget",
        101,
        ["hellaswag", "arc_challenge", "piqa"],
        None,
        "1",
        True,
        merged,
        {"plan": "sha"},
        0.0,
    )
    assert partial["status"] == "general_task_lm_eval_partial"
    assert partial["tasks_remaining"] == ["piqa"]

    completed_raw = module._merge_lm_eval_results(merged, {"results": {"piqa": {"acc,none": 0.7}}})
    completed = module._build_result_report(
        {"utility_scope": "Stage C validation only; never selector objective"},
        "raw_random_equal_budget",
        101,
        ["hellaswag", "arc_challenge", "piqa"],
        None,
        "1",
        True,
        completed_raw,
        {"plan": "sha"},
        0.0,
    )
    assert completed["status"] == "general_task_lm_eval_completed"
    assert completed["tasks_remaining"] == []
    assert module._covers_requested_tasks(completed, ["hellaswag", "arc_challenge", "piqa"])
    assert not module._covers_requested_tasks(completed, ["hellaswag", "arc_challenge", "piqa", "winogrande"])

    suite_partial = module._normalize_suite_status(completed)
    assert suite_partial["status"] == "general_task_lm_eval_partial"
    assert suite_partial["tasks"] == list(module.TASKS)
    assert suite_partial["tasks_remaining"] == ["winogrande"]

    suite_complete = module._normalize_suite_status(
        {
            **completed,
            "lm_eval_results": module._merge_lm_eval_results(
                completed["lm_eval_results"],
                {"results": {"winogrande": {"acc,none": 0.6}}},
            ),
        }
    )
    assert suite_complete["status"] == "general_task_lm_eval_completed"
    assert suite_complete["tasks_completed"] == list(module.TASKS)
    assert suite_complete["tasks_remaining"] == []
    print("[general-task-runner-incremental] partial merge and completion semantics: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
