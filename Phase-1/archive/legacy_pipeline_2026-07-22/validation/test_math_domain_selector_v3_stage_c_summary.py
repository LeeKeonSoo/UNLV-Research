#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_selector_v3_stage_c_summary_report.json"
TABLE_PATH = PROJECT_DIR / "outputs" / "validation" / "math_domain_selector_v3_stage_c_summary_table.md"


def main() -> int:
    subprocess.run(
        [sys.executable, "217_build_math_selector_v3_stage_c_summary.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    report = load_json(REPORT_PATH)
    decision = report["decision"]
    raw = report["arms"]["raw_full_natural"]
    v2 = report["arms"]["curated_math_v2_natural"]
    v3 = report["arms"]["curated_math_v3_natural"]
    assert report["status"] == "math_selector_v3_stage_c_summary_completed"
    assert raw["records"] == 512
    assert v2["records"] == 326
    assert v3["records"] == 367
    assert math.isclose(float(raw["mean_nll"]), 1.4956500481292244, rel_tol=0.0, abs_tol=1e-12)
    assert float(v2["mean_nll"]) > float(raw["mean_nll"])
    assert float(v3["mean_nll"]) < float(v2["mean_nll"])
    assert float(v3["mean_nll"]) > float(raw["mean_nll"])
    assert decision["primary_success"] is False
    assert decision["v3_repairs_v2_failure"] is True
    assert decision["benchmark_guardrail_status"] == "missing_gsm8k_and_math_accuracy_results"
    assert TABLE_PATH.exists()
    print("[math-domain-selector-v3-stage-c-summary] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
