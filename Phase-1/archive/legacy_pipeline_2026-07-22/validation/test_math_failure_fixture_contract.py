#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "validation" / "fixtures" / "math_failure_selector_cases.json"
REPORT_PATH = ROOT / "outputs" / "validation" / "math_failure_fixture_contract_report.json"
CONTRACT_PATH = ROOT / "configs" / "math_domain_selector_v3_redesign_contract.json"


def test_fixture_contract_covers_math_failure_modes() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    categories = {case["category"] for case in cases}
    expected = {
        "long_reasoning_context",
        "multi_step_worked_solution",
        "short_answer_low_context",
        "noisy_extraction_artifact",
        "template_redundancy",
        "coverage_tail_topic",
    }

    assert expected.issubset(categories)
    assert all(case["expected_stage_owner"] in {"stage_a", "stage_b", "stage_c"} for case in cases)
    assert all("heldout_nll" not in " ".join(case["allowed_selector_signals"]) for case in cases)


def test_redesign_contract_forbids_posthoc_math_tuning() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["status"] == "frozen_after_math_v2_failure_before_selector_v3"
    assert contract["forbidden_actions"] == [
        "using heldout math NLL, GSM8K, MATH, or Utility outcomes as Stage-B features",
        "removing failed Math evidence from the paper ledger",
        "claiming all-domain improvement before a new frozen Stage-C pass",
    ]
    assert "retain_all_if_budget_allows" in contract["required_candidate_arms"]
    assert "broader_curated_pool" in contract["required_candidate_arms"]


def test_fixture_report_records_failed_v2_as_boundary_condition() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "math_failure_fixture_contract_ready"
    assert report["math_v2_result"]["decision"] == "failed_stage_c_validation"
    assert report["math_v2_result"]["raw_mean_nll"] == 1.49565
    assert report["math_v2_result"]["curated_mean_nll"] == 1.527065
    assert report["next_selector_allowed"] is False


def main() -> int:
    test_fixture_contract_covers_math_failure_modes()
    test_redesign_contract_forbids_posthoc_math_tuning()
    test_fixture_report_records_failed_v2_as_boundary_condition()
    print("[math-failure-fixture-contract] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
