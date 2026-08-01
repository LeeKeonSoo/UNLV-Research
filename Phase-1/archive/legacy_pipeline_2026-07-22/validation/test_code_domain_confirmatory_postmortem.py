#!/usr/bin/env python3
"""Validate the code-domain confirmatory postmortem contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


REPORT = ROOT / "outputs" / "validation" / "code_domain_confirmatory_postmortem_report.json"
DOC = ROOT / "docs" / "code_domain_confirmatory_postmortem.md"
CONFIRMATORY = ROOT / "outputs" / "validation" / "code_domain_confirmatory_decision_report.json"


def main() -> int:
    report = load_json(REPORT)
    confirmatory = load_json(CONFIRMATORY)

    assert report["status"] == "confirmatory_postmortem_completed"
    assert report["confirmatory_result"]["status"] == "confirmatory_decision_reject_primary_margin_failure"
    assert confirmatory["status"] == report["confirmatory_result"]["status"]
    assert report["decision_implications"]["frozen_confirmatory_result_locked"] is True
    assert report["decision_implications"]["primary_margin_passed"] is False
    assert report["decision_implications"]["directional_stageA_signal_replicated"] is True
    assert report["decision_implications"]["directional_raw_signal_replicated"] is True
    assert report["confirmatory_result"]["gap_to_margin"] > 0
    assert report["effect_shift"]["primary_stageA_minus_curated"]["confirmatory_retention_ratio"] < 1.0
    assert report["heldout_shift"]["repository_jaccard"] == 0.0
    assert report["next_development_cycle"]["new_cycle_required"] is True
    assert report["next_development_cycle"]["must_remain_separate_from_completed_confirmatory_protocol"] is True
    assert "do_not_change_the_frozen_margin_after_confirmatory_outcomes" in report["decision_implications"]["forbidden_response"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert DOC.exists()
    doc_text = DOC.read_text(encoding="utf-8")
    assert "negative primary-margin result" in doc_text
    assert "Utility" in doc_text and "Stage B" in doc_text
    print("[code-domain-confirmatory-postmortem] negative result and next-cycle separation: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
