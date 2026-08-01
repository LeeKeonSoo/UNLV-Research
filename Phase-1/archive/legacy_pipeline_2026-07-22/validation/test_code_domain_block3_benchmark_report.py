#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "code_domain_block3_benchmark_report.json"


def main() -> int:
    report = load_json(REPORT_PATH)

    assert report["status"] == "block3_evalplus_lightweight_benchmark_passed"
    assert report["stage"] == "Stage C validation only"
    assert "Stage-C validation evidence only" in report["utility_scope"]
    assert report["swebench_status"] == "deferred_until_feasibility_gate_passes"

    macros = report["arm_macro_pass_rates"]
    assert macros["curated_v2_equal_budget"] > macros["stageA_random_equal_budget"]
    assert macros["curated_v2_equal_budget"] > macros["raw_random_equal_budget"]

    primary = report["primary_comparison"]
    assert primary["passed"] is True
    assert primary["absolute_macro_pass_rate_delta"] >= primary["required_absolute_macro_pass_rate_improvement"]

    suite_deltas = report["suite_deltas_vs_stageA_random"]
    assert suite_deltas["HumanEval+"] > 0.0
    assert suite_deltas["MBPP+"] > 0.0

    assert report["guardrail_status"] == "evalplus_confirmatory_guardrail_passed"
    assert "EvalPlus HumanEval+ and MBPP+" in report["benchmark_family"]

    print("[code-domain-block3-benchmark-report] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
