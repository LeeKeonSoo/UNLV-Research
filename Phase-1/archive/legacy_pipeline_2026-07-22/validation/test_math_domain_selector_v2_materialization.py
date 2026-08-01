#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_selector_v2_materialization_report.json"


def main() -> int:
    report = load_json(REPORT_PATH)
    assert report["status"] == "math_selector_v2_materialized"
    assert "Stage C validation only" in report["utility_scope"]
    assert report["disjointness"]["curated_stageA_random_disjoint"] is True

    counts = report["stage_counts"]
    assert counts["raw_records"] == 512
    assert counts["stage_a_pass"] > counts["stage_b_v2_selected"] > 0

    curated = report["arms"]["curated_math_v2_equal_budget"]
    style_tokens = curated["style_token_counts"]
    total = int(curated["token_proxy_count"])
    assert total >= report["training_token_budget_cap"]
    assert int(style_tokens.get("proof_or_theorem", 0)) / total <= 0.36
    assert int(style_tokens.get("math_web_text", 0)) / total >= 0.24
    assert int(style_tokens.get("qa_math", 0)) / total >= 0.2

    forbidden = " ".join(report["selector_forbidden_signals"])
    assert "Utility" in forbidden
    assert "GSM8K" in forbidden
    assert "MATH" in forbidden

    print("[math-domain-selector-v2-materialization] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
