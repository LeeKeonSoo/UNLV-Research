#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_natural_budget_v3_freeze_report.json"


def main() -> int:
    subprocess.run([sys.executable, "215_freeze_math_selector_v3_natural_budget.py"], cwd=PROJECT_DIR, check=True)
    report = load_json(REPORT_PATH)
    assert report["status"] == "math_natural_budget_v3_protocol_frozen"
    assert report["training_arms"] == [
        "base_no_update",
        "raw_full_natural",
        "curated_math_v2_natural",
        "curated_math_v3_natural",
    ]
    assert report["arms"]["raw_full_natural"]["records"] == 512
    assert report["arms"]["curated_math_v2_natural"]["records"] == 326
    assert report["arms"]["curated_math_v3_natural"]["records"] == 367
    assert report["v3_vs_v2_repair_checks"]["v3_token_proxy_greater_than_v2"] is True
    assert report["v3_vs_v2_repair_checks"]["v3_proof_or_theorem_tokens_greater_than_v2"] is True
    assert report["v3_vs_raw_natural_budget"]["token_proxy_reduction_fraction"] < 0.1
    assert report["stage_c_outcomes_read"] is False
    assert "Stage C validation only" in report["utility_scope"]
    print("[math-domain-natural-budget-v3-freeze] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
