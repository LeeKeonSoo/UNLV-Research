#!/usr/bin/env python3
"""Build the code-domain EvalPlus guardrail report."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_evalplus_guardrail_report.json"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)
DATASET_LABELS = {"humaneval": "HumanEval+", "mbpp": "MBPP+"}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _parse_result_name(path: Path) -> Tuple[str, str, int | None]:
    stem = path.stem
    if stem.endswith("_eval"):
        stem = stem[:-5]
    dataset, rest = stem.split("_", 1)
    if rest.endswith("_base"):
        return dataset, rest[:-5], None
    match = re.match(r"(.+)_seed(\d+)$", rest)
    if not match:
        raise ValueError(f"Cannot parse EvalPlus result name: {path.name}")
    return dataset, match.group(1), int(match.group(2))


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
    arms = [arm for arm in plan.get("training_arms", []) if arm != "base_no_update"]
    return arms or list(TRAINED_ARMS)


def _stage_label(plan: Dict[str, Any]) -> str:
    return "confirmatory" if "confirmatory_training_recipe" in plan else "development"


def _threshold(plan: Dict[str, Any]) -> float:
    if "external_code_guardrails" in plan:
        return float(plan["external_code_guardrails"]["maximum_allowed_absolute_regression_macro_vs_base"])
    return float(
        plan["stage_c_guardrails"]["evalplus_confirmatory"]["non_inferiority"][
            "maximum_allowed_absolute_regression_macro"
        ]
    )


def _expected_task_count(split: Dict[str, Any], dataset_label: str, stage: str) -> int:
    key = f"{dataset_label}/{stage}"
    summary = split.get("summary", {})
    suite_counts = summary.get("suite_split_counts", {})
    if key in suite_counts:
        return int(suite_counts[key])

    records = split.get("records", [])
    matches = [
        row
        for row in records
        if row.get("dataset_name") == dataset_label and row.get("assigned_split") == stage
    ]
    if matches:
        return len(matches)
    raise KeyError(f"Missing EvalPlus expected task count for {key}; pass the frozen split plan with records/summary.")


def build(plan_path: Path, split_path: Path, output_dir: Path, output_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    split = load_json(split_path)
    stage = _stage_label(plan)
    seeds = _training_seeds(plan)
    trained_arms = _trained_arms(plan)
    threshold = _threshold(plan)
    expected_task_counts = {
        "humaneval": _expected_task_count(split, "HumanEval+", stage),
        "mbpp": _expected_task_count(split, "MBPP+", stage),
    }
    results_dir = output_dir / "evalplus_guardrail" / "results"
    blockers: List[str] = []
    source_sha256 = {
        str(plan_path): sha256_file(plan_path),
        str(split_path): sha256_file(split_path),
    }
    rows: Dict[str, Dict[str, Dict[int | str, Dict[str, Any]]]] = {}
    for path in sorted(results_dir.glob("*_eval.json")) if results_dir.exists() else []:
        dataset, arm, seed = _parse_result_name(path)
        result = load_json(path)
        if result.get("status") != "evalplus_samples_evaluated":
            blockers.append(f"status_mismatch:{path}:{result.get('status')}")
            continue
        rows.setdefault(arm, {}).setdefault(dataset, {})["base" if seed is None else seed] = result
        source_sha256[str(path)] = sha256_file(path)

    required_arms = ["base_no_update", *trained_arms]
    arm_summaries: Dict[str, Any] = {}
    for arm in required_arms:
        datasets = {}
        for dataset in DATASET_LABELS:
            values = rows.get(arm, {}).get(dataset, {})
            expected_task_count = expected_task_counts[dataset]
            if arm == "base_no_update":
                base = values.get("base")
                if base is None:
                    blockers.append(f"missing_evalplus_result:{arm}:{dataset}")
                    continue
                if int(base.get("task_count") or -1) != expected_task_count:
                    blockers.append(
                        f"task_count_mismatch:{arm}:{dataset}:"
                        f"{base.get('task_count')}!={expected_task_count}"
                    )
                    continue
                datasets[DATASET_LABELS[dataset]] = {
                    "pass_rate": float(base["pass_rate"]),
                    "task_count": int(base["task_count"]),
                    "pass_count": int(base["pass_count"]),
                }
            else:
                missing = [
                    seed
                    for seed in seeds
                    if seed not in values
                    or int(values[seed].get("task_count") or -1) != expected_task_count
                ]
                if missing:
                    blockers.append(f"missing_evalplus_result:{arm}:{dataset}:seeds={missing}")
                    continue
                rates = {seed: float(values[seed]["pass_rate"]) for seed in seeds}
                datasets[DATASET_LABELS[dataset]] = {
                    "mean_pass_rate": _mean(rates.values()),
                    "sample_std_pass_rate": _std(rates.values()),
                    "per_seed_pass_rate": {str(seed): rates[seed] for seed in seeds},
                    "task_count": int(values[seeds[0]]["task_count"]),
                }
        if datasets:
            if arm == "base_no_update":
                macro = _mean(row["pass_rate"] for row in datasets.values())
            else:
                macro = _mean(row["mean_pass_rate"] for row in datasets.values())
            arm_summaries[arm] = {"datasets": datasets, "macro_pass_rate": macro}

    comparisons: Dict[str, Any] = {}
    if "base_no_update" in arm_summaries:
        base_macro = float(arm_summaries["base_no_update"]["macro_pass_rate"])
        base_suite = {
            suite: float(row["pass_rate"])
            for suite, row in arm_summaries["base_no_update"]["datasets"].items()
        }
        for arm in trained_arms:
            if arm not in arm_summaries:
                continue
            suite_rows = {}
            suite_pass = True
            for suite, row in arm_summaries[arm]["datasets"].items():
                regression = base_suite[suite] - float(row["mean_pass_rate"])
                passed = regression <= threshold
                suite_pass = suite_pass and passed
                suite_rows[suite] = {
                    "base_pass_rate": base_suite[suite],
                    "arm_mean_pass_rate": float(row["mean_pass_rate"]),
                    "absolute_regression": regression,
                    "passed": passed,
                }
            macro_regression = base_macro - float(arm_summaries[arm]["macro_pass_rate"])
            comparisons[arm] = {
                "macro_base_pass_rate": base_macro,
                "macro_arm_pass_rate": float(arm_summaries[arm]["macro_pass_rate"]),
                "macro_absolute_regression": macro_regression,
                "macro_passed": macro_regression <= threshold,
                "suite_passed": suite_pass,
                "passed": suite_pass and macro_regression <= threshold,
                "suites": suite_rows,
            }

    complete = not blockers and all(arm in comparisons for arm in trained_arms)
    passed = complete and all(row["passed"] for row in comparisons.values())
    pass_status = "evalplus_confirmatory_guardrail_passed" if stage == "confirmatory" else "evalplus_development_guardrail_passed"
    fail_status = "evalplus_confirmatory_guardrail_failed" if stage == "confirmatory" else "evalplus_development_guardrail_failed"
    incomplete_status = "evalplus_confirmatory_guardrail_incomplete" if stage == "confirmatory" else "evalplus_development_guardrail_incomplete"
    report = {
        "schema_version": "code-domain-evalplus-guardrail-report-v1",
        "status": (
            pass_status
            if passed
            else fail_status
            if complete
            else incomplete_status
        ),
        "source_sha256": source_sha256,
        "split_summary": split.get("summary", {}),
        "stage": stage,
        "development_only": stage == "development",
        "arm_summaries": arm_summaries,
        "comparisons_vs_base": comparisons,
        "maximum_allowed_absolute_regression": threshold,
        "blockers": blockers,
        "confirmatory_outcomes_read": stage == "confirmatory",
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "EvalPlus development guardrail only; Stage C evidence, never selector objective.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain EvalPlus guardrail report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.plan, args.split, args.output_dir, args.output)
    print({"status": report["status"], "blockers": report["blockers"][:5]})
    return 0 if not report["status"].endswith("_incomplete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
