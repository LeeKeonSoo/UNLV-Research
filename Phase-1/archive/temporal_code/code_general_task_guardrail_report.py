#!/usr/bin/env python3
"""Build code-domain general-task retention guardrail report."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_RETENTION = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_general_task_guardrail_report.json"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)
TASKS = ("hellaswag", "arc_challenge", "piqa", "winogrande")
PRIMARY_METRICS = {
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "piqa": "acc,none",
    "winogrande": "acc,none",
}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _result_path(output_dir: Path, arm: str, seed: int | None) -> Path:
    if arm == "base_no_update":
        return output_dir / "general_task_guardrail" / "lm_eval" / "base_no_update_base_full.json"
    if seed is None:
        raise ValueError("seed is required for trained-arm general-task results")
    return output_dir / "general_task_guardrail" / "lm_eval" / f"{arm}_seed{seed}_full.json"


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


def _task_metrics(result: Dict[str, Any], task: str) -> Dict[str, float]:
    raw = result["lm_eval_results"]["results"][task]
    metrics = {}
    for key in ("acc,none", "acc_norm,none"):
        if key in raw:
            metrics[key] = float(raw[key])
    return metrics


def build(plan_path: Path, retention_path: Path, output_dir: Path, output_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    retention = load_json(retention_path)
    contract = retention["contract"]["general_task_guardrail"]
    max_suite_regression = float(contract["maximum_allowed_absolute_regression_per_suite"])
    max_macro_regression = float(contract["maximum_allowed_absolute_regression_macro"])
    seeds = _training_seeds(plan)
    arms = _trained_arms(plan)
    stage = _stage_label(plan)
    blockers: List[str] = []
    source_sha256 = {
        str(plan_path): sha256_file(plan_path),
        str(retention_path): sha256_file(retention_path),
    }

    base_path = _result_path(output_dir, "base_no_update", None)
    base_result = None
    if not base_path.exists():
        blockers.append("missing_general_task_lm_eval:base_no_update")
    else:
        base_result = load_json(base_path)
        source_sha256[str(base_path)] = sha256_file(base_path)
        if base_result.get("status") == "general_task_lm_eval_partial":
            blockers.append(
                "partial_general_task_lm_eval:base_no_update:"
                f"remaining={base_result.get('tasks_remaining')}"
            )
        elif base_result.get("status") != "general_task_lm_eval_completed":
            blockers.append(f"status_mismatch:{base_path}:{base_result.get('status')}")

    base_task_scores: Dict[str, float] = {}
    base_diagnostic_scores: Dict[str, Dict[str, float]] = {}
    if base_result is not None:
        for task in TASKS:
            if task not in base_result["lm_eval_results"]["results"]:
                blockers.append(f"missing_base_task:{task}")
                continue
            metrics = _task_metrics(base_result, task)
            primary_metric = PRIMARY_METRICS[task]
            if primary_metric not in metrics:
                blockers.append(f"missing_base_primary_metric:{task}:{primary_metric}")
                continue
            base_task_scores[task] = metrics[primary_metric]
            base_diagnostic_scores[task] = metrics

    arm_summaries: Dict[str, Any] = {}
    comparisons: Dict[str, Any] = {}
    for arm in arms:
        per_seed_task_scores: Dict[int, Dict[str, float]] = {}
        per_seed_diagnostics: Dict[int, Dict[str, Dict[str, float]]] = {}
        for seed in seeds:
            path = _result_path(output_dir, arm, seed)
            if not path.exists():
                blockers.append(f"missing_general_task_lm_eval:{arm}:seed{seed}")
                continue
            row = load_json(path)
            source_sha256[str(path)] = sha256_file(path)
            if row.get("status") == "general_task_lm_eval_partial":
                blockers.append(
                    f"partial_general_task_lm_eval:{arm}:seed{seed}:"
                    f"remaining={row.get('tasks_remaining')}"
                )
                continue
            if row.get("status") != "general_task_lm_eval_completed":
                blockers.append(f"status_mismatch:{path}:{row.get('status')}")
                continue
            if row.get("arm") != arm or int(row.get("seed") or -1) != seed:
                blockers.append(f"identity_mismatch:{path}:{row.get('arm')}:{row.get('seed')}")
                continue
            task_scores: Dict[str, float] = {}
            task_diagnostics: Dict[str, Dict[str, float]] = {}
            for task in TASKS:
                if task not in row["lm_eval_results"]["results"]:
                    blockers.append(f"missing_task:{arm}:seed{seed}:{task}")
                    continue
                metrics = _task_metrics(row, task)
                primary_metric = PRIMARY_METRICS[task]
                if primary_metric not in metrics:
                    blockers.append(f"missing_primary_metric:{arm}:seed{seed}:{task}:{primary_metric}")
                    continue
                task_scores[task] = metrics[primary_metric]
                task_diagnostics[task] = metrics
            if len(task_scores) == len(TASKS):
                per_seed_task_scores[seed] = task_scores
                per_seed_diagnostics[seed] = task_diagnostics
        if len(per_seed_task_scores) != len(seeds):
            continue
        tasks = {}
        for task in TASKS:
            values = [per_seed_task_scores[seed][task] for seed in seeds]
            tasks[task] = {
                "primary_metric": PRIMARY_METRICS[task],
                "mean_score": _mean(values),
                "sample_std_score": _std(values),
                "per_seed_score": {str(seed): per_seed_task_scores[seed][task] for seed in seeds},
                "diagnostic_metrics_by_seed": {
                    str(seed): per_seed_diagnostics[seed][task] for seed in seeds
                },
            }
        macro = _mean(row["mean_score"] for row in tasks.values())
        arm_summaries[arm] = {"tasks": tasks, "macro_score": macro}

        if len(base_task_scores) == len(TASKS):
            task_comparisons = {}
            suite_passed = True
            for task, row in tasks.items():
                regression = base_task_scores[task] - float(row["mean_score"])
                passed = regression <= max_suite_regression
                suite_passed = suite_passed and passed
                task_comparisons[task] = {
                    "base_score": base_task_scores[task],
                    "arm_mean_score": float(row["mean_score"]),
                    "absolute_regression": regression,
                    "maximum_allowed_absolute_regression": max_suite_regression,
                    "passed": passed,
                }
            base_macro = _mean(base_task_scores.values())
            macro_regression = base_macro - macro
            comparisons[arm] = {
                "base_macro_score": base_macro,
                "arm_macro_score": macro,
                "macro_absolute_regression": macro_regression,
                "maximum_allowed_macro_regression": max_macro_regression,
                "macro_passed": macro_regression <= max_macro_regression,
                "suite_passed": suite_passed,
                "passed": suite_passed and macro_regression <= max_macro_regression,
                "tasks": task_comparisons,
            }

    complete = not blockers and len(comparisons) == len(arms)
    passed = complete and all(row["passed"] for row in comparisons.values())
    pass_status = "general_task_confirmatory_guardrail_passed" if stage == "confirmatory" else "general_task_guardrail_passed"
    fail_status = "general_task_confirmatory_guardrail_failed" if stage == "confirmatory" else "general_task_guardrail_failed"
    incomplete_status = "general_task_confirmatory_guardrail_incomplete" if stage == "confirmatory" else "general_task_guardrail_incomplete"
    report = {
        "schema_version": "code-domain-general-task-guardrail-report-v1",
        "status": (
            pass_status
            if passed
            else fail_status
            if complete
            else incomplete_status
        ),
        "source_sha256": source_sha256,
        "retention_contract": contract,
        "primary_metric_policy": {
            "hellaswag": "lm-eval canonical normalized accuracy",
            "arc_challenge": "lm-eval canonical normalized accuracy",
            "piqa": "lm-eval raw accuracy",
            "winogrande": "lm-eval raw accuracy",
            "metric_keys": PRIMARY_METRICS,
            "diagnostic_accuracy_metrics_retained": True,
        },
        "base_no_update": {
            "task_scores": base_task_scores,
            "diagnostic_metrics": base_diagnostic_scores,
            "macro_score": _mean(base_task_scores.values()) if len(base_task_scores) == len(TASKS) else None,
        },
        "arm_summaries": arm_summaries,
        "comparisons_vs_base": comparisons,
        "blockers": blockers,
        "confirmatory_outcomes_read": stage == "confirmatory",
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "General-task retention guardrail only; Stage C evidence, never selector objective.",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain general-task guardrail report.")
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
