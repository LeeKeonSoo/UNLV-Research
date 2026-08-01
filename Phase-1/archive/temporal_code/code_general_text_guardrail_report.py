#!/usr/bin/env python3
"""Build code-domain general-text NLL retention guardrail report."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_RETENTION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_general_text_guardrail_report.json"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _one_sided_upper_95(values: List[float]) -> float:
    if len(values) <= 1:
        return values[0] if values else math.inf
    return _mean(values) + 1.645 * (_std(values) / math.sqrt(len(values)))


def _nll_path(output_dir: Path, arm: str, seed: int | None = None) -> Path:
    name = "base_no_update" if arm == "base_no_update" else f"{arm}_seed{seed}"
    return output_dir / "general_text_guardrail" / "nll" / f"{name}.json"


def _training_recipe(plan: Dict[str, Any]) -> Dict[str, Any]:
    if "training_recipe" in plan:
        return plan["training_recipe"]
    return plan["confirmatory_training_recipe"]


def _training_seeds(plan: Dict[str, Any]) -> List[int]:
    recipe = _training_recipe(plan)
    if "development_training_seeds" in recipe:
        return [int(seed) for seed in recipe["development_training_seeds"]]
    return [int(seed) for seed in recipe["confirmatory_training_seeds"]]


def _trained_arms(plan: Dict[str, Any]) -> List[str]:
    arms = [str(arm) for arm in plan.get("training_arms") or [] if str(arm) != "base_no_update"]
    return arms or list(TRAINED_ARMS)


def _stage_label(plan: Dict[str, Any]) -> str:
    return "confirmatory" if "confirmatory_training_recipe" in plan else "development"


def build(plan_path: Path, retention_path: Path, output_dir: Path, output_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    retention = load_json(retention_path)
    contract = retention["contract"]
    max_increase = float(contract["general_text_guardrail"]["maximum_allowed_mean_nll_increase"])
    seeds = _training_seeds(plan)
    arms = _trained_arms(plan)
    stage = _stage_label(plan)
    blockers: List[str] = []
    source_sha256 = {
        str(plan_path): sha256_file(plan_path),
        str(retention_path): sha256_file(retention_path),
    }
    base_path = _nll_path(output_dir, "base_no_update")
    base_result = None
    if not base_path.exists():
        blockers.append("missing_general_text_nll:base_no_update")
    else:
        base_result = load_json(base_path)
        source_sha256[str(base_path)] = sha256_file(base_path)
        if base_result.get("status") != "general_text_nll_completed":
            blockers.append(f"status_mismatch:{base_path}:{base_result.get('status')}")

    arm_summaries: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    base_nll = float(base_result["mean_nll"]) if base_result else None
    for arm in arms:
        values = {}
        for seed in seeds:
            path = _nll_path(output_dir, arm, seed)
            if not path.exists():
                blockers.append(f"missing_general_text_nll:{arm}:seed{seed}")
                continue
            row = load_json(path)
            source_sha256[str(path)] = sha256_file(path)
            if row.get("status") != "general_text_nll_completed":
                blockers.append(f"status_mismatch:{path}:{row.get('status')}")
                continue
            values[seed] = float(row["mean_nll"])
        if len(values) != len(seeds):
            continue
        ordered = [values[seed] for seed in seeds]
        arm_summaries[arm] = {
            "seeds": seeds,
            "mean_nll": _mean(ordered),
            "sample_std_nll": _std(ordered),
            "per_seed_nll": {str(seed): values[seed] for seed in seeds},
        }
        if base_nll is not None:
            increases = [values[seed] - base_nll for seed in seeds]
            upper = _one_sided_upper_95(increases)
            comparisons[arm] = {
                "base_mean_nll": base_nll,
                "arm_mean_nll": arm_summaries[arm]["mean_nll"],
                "mean_nll_increase_vs_base": _mean(increases),
                "one_sided_95_upper_nll_increase": upper,
                "maximum_allowed_mean_nll_increase": max_increase,
                "passed": upper <= max_increase,
                "per_seed_nll_increase": {str(seed): values[seed] - base_nll for seed in seeds},
            }

    complete = not blockers and len(comparisons) == len(arms)
    passed = complete and all(row["passed"] for row in comparisons.values())
    pass_status = "general_text_confirmatory_guardrail_passed" if stage == "confirmatory" else "general_text_guardrail_passed"
    fail_status = "general_text_confirmatory_guardrail_failed" if stage == "confirmatory" else "general_text_guardrail_failed"
    incomplete_status = "general_text_confirmatory_guardrail_incomplete" if stage == "confirmatory" else "general_text_guardrail_incomplete"
    report = {
        "schema_version": "code-domain-general-text-guardrail-report-v1",
        "status": (
            pass_status
            if passed
            else fail_status
            if complete
            else incomplete_status
        ),
        "source_sha256": source_sha256,
        "retention_contract": contract["general_text_guardrail"],
        "base_no_update": None if base_result is None else {"mean_nll": base_nll, "tokens": base_result.get("tokens")},
        "arm_summaries": arm_summaries,
        "comparisons_vs_base": comparisons,
        "blockers": blockers,
        "confirmatory_outcomes_read": stage == "confirmatory",
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "General-text NLL retention guardrail only; Stage C evidence, never selector objective.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain general-text guardrail report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--retention", type=Path, default=DEFAULT_RETENTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.plan, args.retention, args.output_dir, args.output)
    print({"status": report["status"], "blockers": report["blockers"][:5]})
    return 0 if not report["status"].endswith("_incomplete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
