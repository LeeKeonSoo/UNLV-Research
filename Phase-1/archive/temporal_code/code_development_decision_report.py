#!/usr/bin/env python3
"""Build the frozen code-domain development decision report."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_PLAN_REPORT = OUTPUT_DIR / "validation" / "code_domain_development_plan_qwen3_4b_report.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_development_decision_report.json"
DEFAULT_EVALPLUS_REPORT = OUTPUT_DIR / "validation" / "code_domain_evalplus_guardrail_report.json"
DEFAULT_GENERAL_TEXT_REPORT = OUTPUT_DIR / "validation" / "code_domain_general_text_guardrail_report.json"
DEFAULT_GENERAL_TASK_REPORT = OUTPUT_DIR / "validation" / "code_domain_general_task_guardrail_report.json"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "known_high_quality_equal_budget",
)


def _trained_arms(plan: Dict[str, Any]) -> Tuple[str, ...]:
    arms = tuple(
        str(arm)
        for arm in plan.get("training_arms", TRAINED_ARMS)
        if str(arm) != "base_no_update"
    )
    return arms or TRAINED_ARMS


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _run_result_path(output_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}" / "run_result.json"


def _nll_result_path(output_dir: Path, arm: str, seed: int | None = None) -> Path:
    if arm == "base_no_update":
        return output_dir / "heldout_nll" / "base_no_update.json"
    if seed is None:
        raise ValueError("seed is required for trained-arm NLL results")
    return output_dir / "heldout_nll" / f"{arm}_seed{seed}.json"


def _load_completed_json(path: Path, expected_status: str, blockers: List[str]) -> Dict[str, Any] | None:
    if not path.exists():
        blockers.append(f"missing:{path}")
        return None
    row = load_json(path)
    if row.get("status") != expected_status:
        blockers.append(f"status_mismatch:{path}:{row.get('status')}")
        return None
    return row


def _load_guardrail_report(path: Path, expected_pass_status: str) -> Dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "path": str(path),
            "passed": False,
            "blockers": [f"missing:{path}"],
        }
    row = load_json(path)
    return {
        "status": row.get("status"),
        "path": str(path),
        "passed": row.get("status") == expected_pass_status,
        "blockers": row.get("blockers", []),
        "report": row,
    }


def _arm_summary(values_by_seed: Dict[int, float]) -> Dict[str, Any]:
    values = [values_by_seed[seed] for seed in sorted(values_by_seed)]
    return {
        "seeds": sorted(values_by_seed),
        "mean_nll": _mean(values),
        "sample_std_nll": _sample_std(values),
        "min_nll": min(values),
        "max_nll": max(values),
        "per_seed": {str(seed): values_by_seed[seed] for seed in sorted(values_by_seed)},
    }


def _paired_delta(
    left: Dict[int, float],
    right: Dict[int, float],
    *,
    label: str,
) -> Dict[str, Any]:
    common = sorted(set(left) & set(right))
    deltas = {seed: left[seed] - right[seed] for seed in common}
    values = [deltas[seed] for seed in common]
    return {
        "label": label,
        "interpretation": "positive means left arm has lower NLL than right arm",
        "seeds": common,
        "mean_delta": _mean(values) if values else None,
        "sample_std_delta": _sample_std(values) if values else None,
        "all_seed_deltas_positive": all(value > 0 for value in values) if values else False,
        "per_seed_delta": {str(seed): deltas[seed] for seed in common},
    }


def build(
    plan_path: Path,
    plan_report_path: Path,
    output_dir: Path,
    output_path: Path,
    evalplus_report_path: Path,
    general_text_report_path: Path,
    general_task_report_path: Path,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    plan_report = load_json(plan_report_path)
    recipe = plan["training_recipe"]
    steps = int(recipe["optimizer_steps"])
    seeds = [int(seed) for seed in recipe["development_training_seeds"]]
    trained_arms = _trained_arms(plan)
    margin = float(
        plan.get("practical_effect_margin", {}).get(
            "curated_vs_stageA_random_required_absolute_nll_reduction",
            0.0,
        )
    )
    blockers: List[str] = []
    source_sha256: Dict[str, str] = {
        str(plan_path): sha256_file(plan_path),
        str(plan_report_path): sha256_file(plan_report_path),
    }
    evalplus_guardrail = _load_guardrail_report(
        evalplus_report_path,
        "evalplus_development_guardrail_passed",
    )
    general_text_guardrail = _load_guardrail_report(
        general_text_report_path,
        "general_text_guardrail_passed",
    )
    general_task_guardrail = _load_guardrail_report(
        general_task_report_path,
        "general_task_guardrail_passed",
    )
    if evalplus_report_path.exists():
        source_sha256[str(evalplus_report_path)] = sha256_file(evalplus_report_path)
    if general_text_report_path.exists():
        source_sha256[str(general_text_report_path)] = sha256_file(general_text_report_path)
    if general_task_report_path.exists():
        source_sha256[str(general_task_report_path)] = sha256_file(general_task_report_path)

    train_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for arm in trained_arms:
        for seed in seeds:
            path = _run_result_path(output_dir, arm, seed, steps)
            row = _load_completed_json(path, "development_qlora_completed", blockers)
            if row is None:
                continue
            if int(row.get("optimizer_steps") or -1) != steps:
                blockers.append(f"optimizer_step_mismatch:{arm}:{seed}")
            if int(row.get("seed") or -1) != seed or row.get("arm") != arm:
                blockers.append(f"run_identity_mismatch:{arm}:{seed}")
            train_results[(arm, seed)] = row
            source_sha256[str(path)] = sha256_file(path)

    eval_results: Dict[str, Dict[int, float]] = {arm: {} for arm in trained_arms}
    base_result = _load_completed_json(
        _nll_result_path(output_dir, "base_no_update"),
        "heldout_nll_completed",
        blockers,
    )
    if base_result is not None:
        source_sha256[str(_nll_result_path(output_dir, "base_no_update"))] = sha256_file(
            _nll_result_path(output_dir, "base_no_update")
        )
    for arm in trained_arms:
        for seed in seeds:
            path = _nll_result_path(output_dir, arm, seed)
            row = _load_completed_json(path, "heldout_nll_completed", blockers)
            if row is None:
                continue
            if int(row.get("seed") or -1) != seed or row.get("arm") != arm:
                blockers.append(f"nll_identity_mismatch:{arm}:{seed}")
            eval_results[arm][seed] = float(row["mean_nll"])
            source_sha256[str(path)] = sha256_file(path)

    expected_train = len(trained_arms) * len(seeds)
    expected_eval = expected_train + 1
    completed_eval = sum(len(values) for values in eval_results.values()) + (1 if base_result else 0)
    if len(train_results) != expected_train:
        blockers.append(f"incomplete_training_runs:{len(train_results)}/{expected_train}")
    if completed_eval != expected_eval:
        blockers.append(f"incomplete_heldout_nll:{completed_eval}/{expected_eval}")

    arm_summaries = {
        arm: _arm_summary(values)
        for arm, values in eval_results.items()
        if len(values) == len(seeds)
    }
    primary = plan["primary_comparison"]
    curated = eval_results[primary["treatment"]]
    stage_a = eval_results[primary["primary_baseline"]]
    raw = eval_results["raw_random_equal_budget"]
    reference = eval_results[primary["reference_arm"]]

    nll_gate: Dict[str, Any]
    if blockers:
        nll_gate = {
            "status": "incomplete",
            "blockers": blockers,
        }
        overall_status = "development_decision_incomplete"
    else:
        treatment = primary["treatment"]
        primary_baseline = primary["primary_baseline"]
        reference_arm = primary["reference_arm"]
        curated_mean = arm_summaries[treatment]["mean_nll"]
        stage_a_mean = arm_summaries[primary_baseline]["mean_nll"]
        raw_mean = arm_summaries["raw_random_equal_budget"]["mean_nll"]
        reference_mean = arm_summaries[reference_arm]["mean_nll"]
        primary_reduction = stage_a_mean - curated_mean
        raw_direction_reduction = raw_mean - curated_mean
        reference_gap = reference_mean - curated_mean
        primary_pass = primary_reduction >= margin
        raw_direction_pass = raw_direction_reduction > 0
        nll_gate = {
            "status": "passed" if primary_pass and raw_direction_pass else "failed",
            "primary_margin_required_absolute_nll_reduction": margin,
            "curated_vs_stageA_random_mean_nll_reduction": primary_reduction,
            "curated_vs_stageA_random_margin_pass": primary_pass,
            "curated_vs_raw_random_mean_nll_reduction": raw_direction_reduction,
            "curated_vs_raw_random_direction_pass": raw_direction_pass,
            "known_high_quality_minus_curated_mean_nll": reference_gap,
            "paired_deltas": {
                "stageA_random_minus_curated": _paired_delta(
                    stage_a,
                    curated,
                    label="Stage-A-random minus curated",
                ),
                "raw_random_minus_curated": _paired_delta(
                    raw,
                    curated,
                    label="raw-random minus curated",
                ),
                "known_high_quality_minus_curated": _paired_delta(
                    reference,
                    curated,
                    label="known-high-quality minus curated",
                ),
            },
        }
        required_guardrail_issues = []
        if evalplus_guardrail["status"] == "missing":
            required_guardrail_issues.append("EvalPlus development guardrail evidence is missing")
        elif not evalplus_guardrail["passed"]:
            required_guardrail_issues.append(
                f"EvalPlus development guardrail is not passing: {evalplus_guardrail['status']}"
            )
        if general_text_guardrail["status"] == "missing":
            required_guardrail_issues.append("general-text NLL retention evidence is missing")
        elif not general_text_guardrail["passed"]:
            required_guardrail_issues.append(
                f"general-text NLL retention guardrail is not passing: {general_text_guardrail['status']}"
            )
        if general_task_guardrail["status"] == "missing":
            required_guardrail_issues.append("general-task retention evidence is missing")
        elif not general_task_guardrail["passed"]:
            required_guardrail_issues.append(
                f"general-task retention guardrail is not passing: {general_task_guardrail['status']}"
            )
        if not primary_pass:
            overall_status = "development_decision_do_not_promote_primary_margin_failure"
        elif not raw_direction_pass:
            overall_status = "development_decision_do_not_promote_raw_direction_failure"
        elif (
            evalplus_guardrail["status"] not in ("missing", "evalplus_development_guardrail_incomplete")
            and not evalplus_guardrail["passed"]
        ) or (
            general_text_guardrail["status"] not in ("missing", "general_text_guardrail_incomplete")
            and not general_text_guardrail["passed"]
        ) or (
            general_task_guardrail["status"] not in ("missing", "general_task_guardrail_incomplete")
            and not general_task_guardrail["passed"]
        ):
            overall_status = "development_decision_reject_guardrail_failure"
        elif required_guardrail_issues:
            overall_status = "development_decision_abstain_missing_required_guardrails"
        else:
            overall_status = "development_decision_promote_to_confirmatory"
        nll_gate["required_guardrail_issues"] = required_guardrail_issues

    retention_guardrails = plan["general_retention_guardrails"]
    general_task_guardrail_plan = retention_guardrails.get(
        "general_task_guardrail",
        retention_guardrails,
    )
    retention_decision_rule = retention_guardrails.get("decision_rule", retention_guardrails)

    report = {
        "schema_version": "code-domain-development-decision-report-v1",
        "status": overall_status,
        "source_sha256": source_sha256,
        "summary": {
            "training_runs_completed": len(train_results),
            "expected_training_runs": expected_train,
            "heldout_nll_results_completed": completed_eval,
            "expected_heldout_nll_results": expected_eval,
            "base_no_update_mean_nll": None if base_result is None else float(base_result["mean_nll"]),
            "arm_summaries": arm_summaries,
            "nll_gate": nll_gate,
            "stage_c_guardrails": {
                "evalplus_development": {
                    "status": evalplus_guardrail["status"],
                    "path": evalplus_guardrail["path"],
                    "passed": evalplus_guardrail["passed"],
                    "blocker_count": len(evalplus_guardrail["blockers"]),
                    "blockers": evalplus_guardrail["blockers"][:10],
                },
                "general_text_nll_retention": {
                    "status": general_text_guardrail["status"],
                    "path": general_text_guardrail["path"],
                    "passed": general_text_guardrail["passed"],
                    "blocker_count": len(general_text_guardrail["blockers"]),
                    "blockers": general_text_guardrail["blockers"][:10],
                },
                "general_task_retention": {
                    "status": general_task_guardrail["status"],
                    "path": general_task_guardrail["path"],
                    "suites": general_task_guardrail_plan.get("suites", []),
                    "missing_evidence_action": retention_decision_rule.get(
                        "missing_evidence_action",
                        "abstain",
                    ),
                    "passed": general_task_guardrail["passed"],
                    "blocker_count": len(general_task_guardrail["blockers"]),
                    "blockers": general_task_guardrail["blockers"][:10],
                },
            },
            "development_decision_rule": plan["development_decision_rule"],
            "external_code_guardrails": plan["external_code_guardrails"],
            "general_retention_guardrails": plan["general_retention_guardrails"],
            "blockers": blockers,
        },
        "plan_freeze_summary": plan_report.get("summary", {}),
        "confirmatory_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": (
            "Development heldout NLL decision report only. NLL margin can support a "
            "development-stage recipe decision, but promotion remains abstained until "
            "frozen guardrail evidence is present and passing."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain development decision report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--plan-report", type=Path, default=DEFAULT_PLAN_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evalplus-report", type=Path, default=DEFAULT_EVALPLUS_REPORT)
    parser.add_argument("--general-text-report", type=Path, default=DEFAULT_GENERAL_TEXT_REPORT)
    parser.add_argument("--general-task-report", type=Path, default=DEFAULT_GENERAL_TASK_REPORT)
    args = parser.parse_args()
    report = build(
        args.plan,
        args.plan_report,
        args.output_dir,
        args.output,
        args.evalplus_report,
        args.general_text_report,
        args.general_task_report,
    )
    print(
        {
            "status": report["status"],
            "training_runs_completed": report["summary"]["training_runs_completed"],
            "heldout_nll_results_completed": report["summary"]["heldout_nll_results_completed"],
            "nll_gate_status": report["summary"]["nll_gate"]["status"],
        }
    )
    return 0 if report["status"] != "development_decision_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
