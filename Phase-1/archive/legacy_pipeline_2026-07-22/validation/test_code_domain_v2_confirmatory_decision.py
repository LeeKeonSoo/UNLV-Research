#!/usr/bin/env python3
"""Contract checks for the v2 confirmatory decision report."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "code_domain_v2_confirmatory_protocol_qwen3_4b.json")
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json"
    )
    summary = report["summary"]
    nll_gate = summary["nll_gate"]

    assert report["schema_version"] == "code-domain-v2-confirmatory-decision-report-v1"
    assert report["confirmatory_outcomes_read"] is True
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert summary["training_runs_completed"] == summary["expected_training_runs"] == 20
    assert summary["heldout_nll_results_completed"] == summary["expected_heldout_nll_results"] == 21
    assert summary["blockers"] == []

    assert nll_gate["primary_margin_required_absolute_nll_reduction"] == 0.003
    assert nll_gate["curated_vs_stageA_random_margin_pass"] is True
    assert nll_gate["curated_vs_stageA_random_all_paired_seed_pass"] is True
    assert nll_gate["curated_vs_raw_random_direction_pass"] is True
    assert nll_gate["status"] == "passed"
    assert report["status"] == "v2_confirmatory_decision_passed"
    assert (
        nll_gate["curated_vs_stageA_random_mean_nll_reduction"]
        >= protocol["primary_success_rule"]["required_absolute_nll_reduction"]
    )

    assert set(summary["arm_summaries"]) == {
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_v2_equal_budget",
        "known_high_quality_equal_budget",
    }
    assert summary["base_no_update_mean_nll"] is not None
    assert protocol["confirmatory_outcomes_read"] is False
    assert report["protocol_freeze_summary"]["primary_success_rule"]["required_absolute_nll_reduction"] == 0.003

    guardrails = summary["stage_c_guardrails"]
    assert guardrails["evalplus_confirmatory"]["evidence_state"] == "passed"
    assert guardrails["general_task_retention"]["evidence_state"] == "passed"
    assert guardrails["general_text_nll_retention"]["evidence_state"] == "passed"

    print("[code-domain-v2-confirmatory-decision] frozen NLL decision contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
