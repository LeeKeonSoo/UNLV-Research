#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


FREEZE_PATH = Path("configs") / "code_domain_block3_benchmark_execution_freeze_v1.json"
EVALPLUS_REPORT_PATH = OUTPUT_DIR / "validation" / "code_domain_v2_evalplus_confirmatory_guardrail_report.json"
OUTPUT_PATH = OUTPUT_DIR / "validation" / "code_domain_block3_benchmark_report.json"


def _arm_macro(report: Dict[str, Any], arm: str) -> float:
    return float(report["arm_summaries"][arm]["macro_pass_rate"])


def _suite_mean(report: Dict[str, Any], arm: str, suite: str) -> float:
    row = report["arm_summaries"][arm]["datasets"][suite]
    if "mean_pass_rate" in row:
        return float(row["mean_pass_rate"])
    return float(row["pass_rate"])


def build() -> Dict[str, Any]:
    freeze = load_json(FREEZE_PATH)
    evalplus = load_json(EVALPLUS_REPORT_PATH)
    rule = freeze["benchmark_tiers"][1]["success_rule"]
    margin = float(rule["required_absolute_macro_pass_rate_improvement"])

    curated = _arm_macro(evalplus, "curated_v2_equal_budget")
    stage_a = _arm_macro(evalplus, "stageA_random_equal_budget")
    raw = _arm_macro(evalplus, "raw_random_equal_budget")
    reference = _arm_macro(evalplus, "known_high_quality_equal_budget")
    delta_vs_stage_a = curated - stage_a
    delta_vs_raw = curated - raw
    passed = evalplus["status"] == "evalplus_confirmatory_guardrail_passed" and delta_vs_stage_a >= margin

    report = {
        "schema_version": "code-domain-block3-benchmark-report-v1",
        "status": "block3_evalplus_lightweight_benchmark_passed" if passed else "block3_evalplus_lightweight_benchmark_inconclusive",
        "block3_decision": freeze["decision"],
        "utility_scope": freeze["utility_scope"],
        "benchmark_family": "EvalPlus HumanEval+ and MBPP+",
        "stage": "Stage C validation only",
        "primary_rule": rule,
        "arm_macro_pass_rates": {
            "raw_random_equal_budget": raw,
            "stageA_random_equal_budget": stage_a,
            "curated_v2_equal_budget": curated,
            "known_high_quality_equal_budget": reference,
        },
        "primary_comparison": {
            "treatment": "curated_v2_equal_budget",
            "baseline": "stageA_random_equal_budget",
            "absolute_macro_pass_rate_delta": delta_vs_stage_a,
            "required_absolute_macro_pass_rate_improvement": margin,
            "passed": delta_vs_stage_a >= margin,
        },
        "supporting_comparison": {
            "baseline": "raw_random_equal_budget",
            "absolute_macro_pass_rate_delta": delta_vs_raw,
            "passed": delta_vs_raw > 0.0,
        },
        "suite_deltas_vs_stageA_random": {
            suite: _suite_mean(evalplus, "curated_v2_equal_budget", suite)
            - _suite_mean(evalplus, "stageA_random_equal_budget", suite)
            for suite in ("HumanEval+", "MBPP+")
        },
        "guardrail_status": evalplus["status"],
        "swebench_status": "deferred_until_feasibility_gate_passes",
        "claim_boundary": freeze["claim_boundary"]["block3_additional_claim_if_passed"] if passed else rule["inconclusive_action"],
        "source_sha256": {
            str(FREEZE_PATH): sha256_file(FREEZE_PATH),
            str(EVALPLUS_REPORT_PATH): sha256_file(EVALPLUS_REPORT_PATH),
        },
    }
    save_json(OUTPUT_PATH, report)
    return report


def main() -> int:
    report = build()
    print(f"[code-domain-block3-benchmark] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
