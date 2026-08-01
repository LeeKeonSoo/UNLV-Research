#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_equal_token_arms_report.json"


def main() -> int:
    report = load_json(REPORT_PATH)
    assert report["status"] == "math_equal_token_arms_materialized"
    assert "Stage C validation only" in report["utility_scope"]
    assert report["training_token_budget_cap"] >= 100000
    assert report["disjointness"]["curated_stageA_random_disjoint"] is True

    counts = report["stage_counts"]
    assert counts["raw_records"] == 512
    assert counts["stage0_retained"] > 0
    assert counts["stage_a_pass"] > counts["stage_b_selected"] > 0
    assert counts["reference_stage_a_pass"] > 0

    arms = report["arms"]
    expected = {
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_math_equal_budget",
        "known_high_quality_equal_budget",
    }
    assert set(arms) == expected
    for arm in expected:
        assert arms[arm]["records"] > 0
        assert arms[arm]["token_proxy_count"] >= report["training_token_budget_cap"]

    forbidden = " ".join(report["selector_forbidden_signals"])
    assert "GSM8K" in forbidden
    assert "MATH" in forbidden
    assert "Utility" in forbidden

    print("[math-domain-equal-token-materialization] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
