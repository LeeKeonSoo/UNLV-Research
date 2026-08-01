#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
PLAN_PATH = Path("configs") / "math_domain_natural_budget_v3_protocol_qwen3_4b.json"
RUN_DIR = OUTPUT_DIR / "math_domain_natural_budget_v3_qwen3_4b"
OUTPUT_PATH = OUTPUT_DIR / "validation" / "math_domain_selector_v3_stage_c_summary_report.json"
TABLE_PATH = OUTPUT_DIR / "validation" / "math_domain_selector_v3_stage_c_summary_table.md"


@dataclass(frozen=True, slots=True)
class ArmSummaryInputs:
    freeze: JsonMap
    blocks: JsonMap
    plan: JsonMap


@dataclass(frozen=True, slots=True)
class StageCComparison:
    plan: JsonMap
    raw_nll: JsonMap
    v2_nll: JsonMap
    v3_nll: JsonMap


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
    return {
        "mean_nll": _mean(series),
        "sample_std_nll": _std(series),
        "per_seed_mean_nll": values,
    }


def _arm_size(inputs: ArmSummaryInputs, arm: str) -> JsonMap:
    block_row = inputs.blocks["blocks"][arm]
    return {
        "records": int(inputs.freeze["arms"][arm]["records"]),
        "token_proxy_count": int(inputs.freeze["arms"][arm]["token_proxy_count"]),
        "packed_training_tokens": int(block_row["tokens_in_blocks"]),
        "optimizer_steps": int(inputs.plan["confirmatory_training_recipe"]["optimizer_steps_by_arm"][arm]),
        "style_token_counts": inputs.freeze["arms"][arm]["style_token_counts"],
    }


def _reduction(raw_size: JsonMap, curated_size: JsonMap) -> JsonMap:
    return {
        "record_reduction_fraction": 1.0 - (curated_size["records"] / raw_size["records"]),
        "token_proxy_reduction_fraction": 1.0 - (curated_size["token_proxy_count"] / raw_size["token_proxy_count"]),
        "packed_training_token_reduction_fraction": 1.0
        - (curated_size["packed_training_tokens"] / raw_size["packed_training_tokens"]),
        "optimizer_step_reduction_fraction": 1.0 - (curated_size["optimizer_steps"] / raw_size["optimizer_steps"]),
    }


def _arm_summary(size: JsonMap, nll: JsonMap) -> JsonMap:
    return {**size, **nll}


def _decision(comparison: StageCComparison) -> JsonMap:
    required = float(comparison.plan["primary_success_rule"]["required_absolute_nll_reduction"])
    raw_mean = float(comparison.raw_nll["mean_nll"])
    v2_mean = float(comparison.v2_nll["mean_nll"])
    v3_mean = float(comparison.v3_nll["mean_nll"])
    v3_beats_raw_by_margin = v3_mean <= raw_mean - required
    v3_repairs_v2 = v3_mean < v2_mean
    if v3_beats_raw_by_margin:
        label = "math_selector_v3_primary_nll_success"
    elif v3_repairs_v2:
        label = "math_selector_v3_repairs_v2_but_does_not_beat_raw_full_natural_budget"
    else:
        label = "math_selector_v3_failed_to_repair_v2"
    return {
        "label": label,
        "primary_success": v3_beats_raw_by_margin,
        "v3_repairs_v2_failure": v3_repairs_v2,
        "required_absolute_nll_reduction": required,
        "v3_minus_raw_mean_nll_lower_is_better": v3_mean - raw_mean,
        "v3_minus_v2_mean_nll_lower_is_better": v3_mean - v2_mean,
        "benchmark_guardrail_status": "missing_gsm8k_and_math_accuracy_results",
        "paper_claim_allowed": "abstain_for_math_domain_primary_success",
    }


def _write_table(report: JsonMap) -> None:
    arms = report["arms"]
    rows = [
        ("Base", "base_no_update"),
        ("Raw", "raw_full_natural"),
        ("Curated v2", "curated_math_v2_natural"),
        ("Curated v3", "curated_math_v3_natural"),
    ]
    lines = [
        "| Arm | Records | Token proxy | Packed train tokens | Optimizer steps | NLL mean | NLL std | Raw delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    raw_mean = float(arms["raw_full_natural"]["mean_nll"])
    for label, arm in rows:
        row = arms[arm]
        records = row.get("records", "-")
        token_proxy = row.get("token_proxy_count", "-")
        packed_tokens = row.get("packed_training_tokens", "-")
        optimizer_steps = row.get("optimizer_steps", 0)
        mean_nll = float(row["mean_nll"])
        std_nll = float(row.get("sample_std_nll", 0.0))
        raw_delta = mean_nll - raw_mean
        lines.append(
            f"| {label} | {records} | {token_proxy} | {packed_tokens} | {optimizer_steps} | "
            f"{mean_nll:.6f} | {std_nll:.6f} | {raw_delta:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Decision: `{report['decision']['label']}`.",
            "Lower NLL is better. Math benchmark guardrails are still missing.",
        ]
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build() -> JsonMap:
    plan = load_json(PLAN_PATH)
    seeds = _seeds(plan)
    freeze = load_json(RUN_DIR / "natural_budget_v3_freeze_report.json")
    blocks = load_json(RUN_DIR / "token_blocks" / "block_manifest.json")
    arm_inputs = ArmSummaryInputs(freeze=freeze, blocks=blocks, plan=plan)
    nll_dir = RUN_DIR / "heldout_nll"
    base_nll = load_json(nll_dir / "base_no_update.json")
    raw_size = _arm_size(arm_inputs, "raw_full_natural")
    v2_size = _arm_size(arm_inputs, "curated_math_v2_natural")
    v3_size = _arm_size(arm_inputs, "curated_math_v3_natural")
    raw_nll = _nll_arm(nll_dir, "raw_full_natural", seeds)
    v2_nll = _nll_arm(nll_dir, "curated_math_v2_natural", seeds)
    v3_nll = _nll_arm(nll_dir, "curated_math_v3_natural", seeds)
    comparison = StageCComparison(plan=plan, raw_nll=raw_nll, v2_nll=v2_nll, v3_nll=v3_nll)
    report = {
        "schema_version": "math-domain-selector-v3-stage-c-summary-v1",
        "status": "math_selector_v3_stage_c_summary_completed",
        "seed_scope": seeds,
        "arms": {
            "base_no_update": {
                "mean_nll": float(base_nll["mean_nll"]),
                "eval_tokens": int(base_nll["tokens"]),
                "optimizer_steps": 0,
            },
            "raw_full_natural": _arm_summary(raw_size, raw_nll),
            "curated_math_v2_natural": _arm_summary(v2_size, v2_nll),
            "curated_math_v3_natural": _arm_summary(v3_size, v3_nll),
        },
        "natural_budget_reduction_v3_vs_raw": _reduction(raw_size, v3_size),
        "natural_budget_reduction_v2_vs_raw": _reduction(raw_size, v2_size),
        "decision": _decision(comparison),
        "source_sha256": {
            str(PLAN_PATH): sha256_file(PLAN_PATH),
            str(RUN_DIR / "natural_budget_v3_freeze_report.json"): sha256_file(
                RUN_DIR / "natural_budget_v3_freeze_report.json"
            ),
            str(RUN_DIR / "token_blocks" / "block_manifest.json"): sha256_file(
                RUN_DIR / "token_blocks" / "block_manifest.json"
            ),
            str(nll_dir / "evaluate_missing_summary.json"): sha256_file(nll_dir / "evaluate_missing_summary.json"),
        },
        "utility_scope": plan["utility_scope"],
        "claim_boundary": (
            "Math selector v3 Stage-C NLL summary only. It diagnoses v2 repair but does not support "
            "a math-domain success or production-release claim without benchmark guardrails."
        ),
    }
    save_json(OUTPUT_PATH, report)
    _write_table(report)
    return report


def main() -> int:
    print(json.dumps(build(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
