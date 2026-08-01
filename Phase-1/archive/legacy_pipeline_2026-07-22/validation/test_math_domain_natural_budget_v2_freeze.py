#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_natural_budget_v2_freeze_report.json"


def main() -> int:
    report = load_json(REPORT_PATH)
    assert report["status"] == "math_natural_budget_v2_protocol_frozen"
    assert report["training_arms"] == ["base_no_update", "raw_full_natural", "curated_math_v2_natural"]
    assert report["arms"]["raw_full_natural"]["records"] == 512
    assert report["arms"]["curated_math_v2_natural"]["records"] == 326
    assert report["natural_budget_reduction_curated_v2_vs_raw"]["token_proxy_reduction_fraction"] > 0.4
    assert "Stage C validation only" in report["utility_scope"]
    print("[math-domain-natural-budget-v2-freeze] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
