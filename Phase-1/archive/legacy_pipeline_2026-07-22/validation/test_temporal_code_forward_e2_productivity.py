#!/usr/bin/env python3
"""Contract checks for the forward E2 productivity report."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_forward_e2_productivity_report.json"
    )
    observed = report["observed_pilot"]
    estimates = report["point_estimate_only"]
    interpretation = report["interpretation"]
    assert report["status"] == "forward_e2_acquisition_feasible_but_not_ready_for_utility"
    assert observed["metadata_candidate_count"] == 16
    assert observed["execution_candidate_count"] == 5
    assert observed["task_valid_e2_count"] == 2
    assert observed["failure_stage_counts"]["task_valid_e2"] == 2
    assert estimates["metadata_candidates_needed_for_1083"] == 8664
    assert estimates["execution_attempts_needed_for_1083"] == 2708
    assert interpretation["pilot_tasks_evaluation_authorized"] is False
    assert interpretation["pilot_too_small_for_capacity_commitment"] is True
    assert interpretation["inferential_yield_or_capacity_claim_allowed"] is False
    assert interpretation["development_utility_may_start"] is False
    assert interpretation["confirmatory_outcomes_read"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-forward-e2] productivity estimate remains planning-only: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
