#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
PLAN_PATH = Path("configs") / "code_domain_natural_budget_protocol_qwen3_4b_v1.json"
RUN_DIR = OUTPUT_DIR / "code_domain_natural_budget_qwen3_4b"
EVALPLUS_REPORT = OUTPUT_DIR / "validation" / "code_domain_natural_budget_evalplus_guardrail_report.json"
OUTPUT_PATH = OUTPUT_DIR / "validation" / "code_domain_natural_budget_stage_c_summary_report.json"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _seeds(plan: JsonMap) -> list[int]:
    return [int(seed) for seed in plan["confirmatory_training_recipe"]["confirmatory_training_seeds"]]


def _nll_arm(nll_dir: Path, arm: str, seeds: list[int]) -> JsonMap:
    values = {}
    for seed in seeds:
        row = load_json(nll_dir / f"{arm}_seed{seed}.json")
        values[str(seed)] = float(row["mean_nll"])
    series = list(values.values())
    return {"mean_nll": _mean(series), "sample_std_nll": _std(series), "per_seed_mean_nll": values}


def _arm_size(arms: JsonMap, steps: JsonMap, arm: str) -> JsonMap:
    return {
        "records": int(arms["arms"][arm]["records"]),
        "token_proxy_count": int(arms["arms"][arm]["token_proxy_count"]),
        "packed_training_tokens": int(steps["packed_tokens_by_arm"][arm]),
        "optimizer_steps": int(steps["optimizer_steps_by_arm"][arm]),
    }


def _evalplus_arm(report: JsonMap, arm: str) -> JsonMap:
    row = report["arm_summaries"][arm]
    suites = {}
    for suite, suite_row in row["datasets"].items():
        suites[suite] = {
            "mean_pass_rate": float(suite_row["mean_pass_rate"]),
            "sample_std_pass_rate": float(suite_row["sample_std_pass_rate"]),
            "per_seed_pass_rate": suite_row["per_seed_pass_rate"],
        }
    return {"macro_pass_rate": float(row["macro_pass_rate"]), "suites": suites}


def build(
    plan_path: Path = PLAN_PATH,
    data_dir: Path = RUN_DIR,
    nll_dir: Path | None = None,
    evalplus_report: Path = EVALPLUS_REPORT,
    output_path: Path = OUTPUT_PATH,
) -> JsonMap:
    plan = load_json(plan_path)
    seeds = _seeds(plan)
    arms_path = data_dir / "natural_budget_arms_report.json"
    steps_path = data_dir / "token_blocks" / "natural_budget_steps_report.json"
    effective_nll_dir = nll_dir or data_dir / "heldout_nll"
    arms = load_json(arms_path)
    steps = load_json(steps_path)
    evalplus = load_json(evalplus_report)
    base_nll_path = effective_nll_dir / "base_no_update.json"
    base_nll = load_json(base_nll_path)
    raw_nll = _nll_arm(effective_nll_dir, "raw_full_natural", seeds)
    curated_nll = _nll_arm(effective_nll_dir, "curated_v2_natural", seeds)
    raw_evalplus = _evalplus_arm(evalplus, "raw_full_natural")
    curated_evalplus = _evalplus_arm(evalplus, "curated_v2_natural")
    raw_size = _arm_size(arms, steps, "raw_full_natural")
    curated_size = _arm_size(arms, steps, "curated_v2_natural")
    report = {
        "schema_version": "code-domain-natural-budget-stage-c-summary-v1",
        "status": "code_natural_budget_stage_c_summary_completed",
        "seed_scope": seeds,
        "arms": {
            "base_no_update": {"mean_nll": float(base_nll["mean_nll"]), "eval_tokens": int(base_nll["tokens"])},
            "raw_full_natural": {**raw_size, **raw_nll, "evalplus": raw_evalplus},
            "curated_v2_natural": {**curated_size, **curated_nll, "evalplus": curated_evalplus},
        },
        "natural_budget_reduction_curated_vs_raw": {
            "record_reduction_fraction": 1.0 - (curated_size["records"] / raw_size["records"]),
            "token_proxy_reduction_fraction": 1.0 - (curated_size["token_proxy_count"] / raw_size["token_proxy_count"]),
            "packed_training_token_reduction_fraction": 1.0 - (curated_size["packed_training_tokens"] / raw_size["packed_training_tokens"]),
            "optimizer_step_reduction_fraction": 1.0 - (curated_size["optimizer_steps"] / raw_size["optimizer_steps"]),
        },
        "deltas_curated_minus_raw": {
            "mean_nll_lower_is_better": curated_nll["mean_nll"] - raw_nll["mean_nll"],
            "evalplus_macro_pass_rate_higher_is_better": curated_evalplus["macro_pass_rate"] - raw_evalplus["macro_pass_rate"],
        },
        "decision": (
            "curated_better_than_raw_full_on_nll_and_evalplus"
            if curated_nll["mean_nll"] < raw_nll["mean_nll"]
            and curated_evalplus["macro_pass_rate"] > raw_evalplus["macro_pass_rate"]
            else "mixed_or_failed_natural_budget_stage_c"
        ),
        "source_sha256": {
            str(path): sha256_file(path)
            for path in [
                plan_path,
                evalplus_report,
                arms_path,
                steps_path,
                base_nll_path,
                *[
                    effective_nll_dir / f"{arm}_seed{seed}.json"
                    for arm in ("raw_full_natural", "curated_v2_natural")
                    for seed in seeds
                ],
            ]
        },
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "Code natural-budget Stage-C summary only; does not validate math selector v2 or production release.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the code natural-budget Stage-C summary.")
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--data-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--nll-dir", type=Path)
    parser.add_argument("--evalplus-report", type=Path, default=EVALPLUS_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    report = build(
        plan_path=args.plan,
        data_dir=args.data_dir,
        nll_dir=args.nll_dir,
        evalplus_report=args.evalplus_report,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
