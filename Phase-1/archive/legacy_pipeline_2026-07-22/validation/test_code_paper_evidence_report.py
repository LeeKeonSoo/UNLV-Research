#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "validation" / "code_paper_evidence_report.json"
MD_REPORT_PATH = ROOT / "outputs" / "validation" / "code_paper_evidence_report.md"


def test_code_evidence_report_contains_nll_evalplus_and_budget() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "code_paper_evidence_ready"
    assert report["claim"] == "code_positive_natural_budget_stage_c"
    assert report["framework_compatibility"]["current_artifacts_match"] is True
    assert report["framework_compatibility"]["missing_frozen_implementation_hashes"] == []
    assert report["framework_compatibility"]["mismatched_implementation_hashes"] == []
    assert report["nll"]["result"] == "pass"
    assert report["paper_table_row"]["decision"] == "pass"
    assert report["protocol_id"] == "code-domain-natural-budget-stage-c-summary-v1"
    assert report["protocol_lineage"]["mixed_protocol_values_forbidden"] is True
    assert report["nll"]["curated_minus_raw"] < 0
    assert report["nll"]["packed_token_reduction_fraction"] > 0.6
    assert report["evalplus"]["curated_macro_pass_rate"] > report["evalplus"]["raw_macro_pass_rate"]
    assert report["evalplus"]["evaluation_scope"] == "natural_budget_same_arms_same_seed_scope"
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    markdown = MD_REPORT_PATH.read_text(encoding="utf-8")
    assert "Status: `code_paper_evidence_ready`" in markdown
    assert "Decision: `pass`" in markdown


def main() -> int:
    test_code_evidence_report_contains_nll_evalplus_and_budget()
    print("[code-paper-evidence] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
