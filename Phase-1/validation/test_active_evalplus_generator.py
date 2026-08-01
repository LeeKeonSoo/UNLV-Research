#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.evalplus_generator import adapter_directory, task_ids_for_scope, task_ids_for_split, trim_completion


def main() -> int:
    records = [
        {"dataset": "HumanEval+", "task_id": "HumanEval/2", "assigned_split": "development"},
        {"dataset": "HumanEval+", "task_id": "HumanEval/1", "assigned_split": "development"},
        {"dataset": "HumanEval+", "task_id": "HumanEval/3", "assigned_split": "confirmatory"},
    ]

    assert task_ids_for_split(records, "HumanEval+", "development") == ["HumanEval/1", "HumanEval/2"]
    all_tasks = {"HumanEval/1": {}, "HumanEval/2": {}, "HumanEval/3": {}}
    assert task_ids_for_scope(records, all_tasks, "HumanEval+", "official_full") == [
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/3",
    ]
    assert adapter_directory(Path("D:/runs"), "raw_safe_natural", 23, 429) == Path(
        "D:/runs/qlora_runs/raw_safe_natural_seed23_steps429"
    )
    assert trim_completion("```python\nreturn x\n```\n# Task next") == "return x\n"
    print("[active-evalplus-generator] pure contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
