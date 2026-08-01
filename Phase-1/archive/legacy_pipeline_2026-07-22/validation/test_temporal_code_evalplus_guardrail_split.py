#!/usr/bin/env python3
"""Contract checks for frozen EvalPlus guardrail split."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    plan = load_json(
        PROJECT_DIR / "outputs" / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
    )
    contract = plan["contract"]
    records = plan["records"]
    assert plan["status"] == "frozen_e2_guardrail_split_before_model_outcomes"
    assert plan["summary"]["task_count"] == 542
    assert set(plan["summary"]["split_counts"]) == {"development", "confirmatory"}
    assert all(row["assigned_split"] in {"development", "confirmatory"} for row in records)
    assert len({(row["dataset"], row["task_id"]) for row in records}) == len(records)
    assert contract["aggregate"]["primary_guardrail_aggregate"] == (
        "unweighted macro mean of HumanEval+ and MBPP+ pass_at_1"
    )
    assert contract["non_inferiority"]["maximum_allowed_absolute_regression_macro"] == 0.02
    assert len(contract["development_training_seeds"]) == 5
    assert len(contract["confirmatory_training_seeds"]) == 5
    assert plan["summary"]["task_content_persisted"] is False
    assert plan["summary"]["model_outcomes_read"] is False
    assert plan["development_utility_may_start"] is False
    assert plan["confirmatory_outcomes_read"] is False
    print("[temporal-code-evalplus-split] E2 external guardrail split frozen outcome-free: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
