#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "code_domain_natural_budget_stage_c_summary_report.json"


def main() -> int:
    report = load_json(REPORT_PATH)
    assert report["status"] == "code_natural_budget_stage_c_summary_completed"
    assert report["seed_scope"] == [101, 131, 163, 197, 239]
    assert report["decision"] == "curated_better_than_raw_full_on_nll_and_evalplus"
    assert report["deltas_curated_minus_raw"]["mean_nll_lower_is_better"] < 0.0
    assert report["deltas_curated_minus_raw"]["evalplus_macro_pass_rate_higher_is_better"] > 0.0
    assert report["natural_budget_reduction_curated_vs_raw"]["token_proxy_reduction_fraction"] > 0.5
    assert any("current_framework_rerun" in path for path in report["source_sha256"])
    assert "Stage C validation only" in report["utility_scope"]
    print("[code-domain-natural-budget-stage-c-summary] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
