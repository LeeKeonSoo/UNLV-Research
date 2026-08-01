#!/usr/bin/env python3
"""Validate the frozen redundancy proxy candidate decision."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json


REPORT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_decision_report.json"
)


def main() -> int:
    report = load_json(REPORT)
    assert report["status"] == "redundancy_proxy_candidate_decision_frozen"
    assert not report["blockers"]
    assert report["candidate"] == "log_count"
    assert report["canonical_control"] == "binary_current"
    assert report["candidate_decision"] == (
        "hold_log_count_keep_binary_current_directional_nonworse_failed"
    )
    assert report["promotion_allowed"] is False
    assert report["qwen3_4b_development_allowed"] is False
    assert report["curation_effect"]["passed"] is True
    comparison = report["candidate_vs_binary"]
    assert comparison["statistical_noninferiority_passed"] is True
    assert comparison["directional_nonworse_passed"] is False
    assert comparison["nonpositive_seed_count"] == 0
    assert comparison["mean"] > 0
    assert comparison["mean"] < comparison["maximum_upper_bound"]
    assert report["promotion_requirements"][
        "template_saturation_mechanism_precheck"
    ] is True
    assert report["promotion_requirements"]["general_text_retention"] is True
    assert report["futility_rule"]["triggered"] is True
    assert report["framework_evidence"]["release_status"].startswith("abstain_")
    assert "keep binary_current canonical" in report["required_next_work"]
    print("[redundancy-proxy-decision] hold log_count, retain binary, release abstain: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
