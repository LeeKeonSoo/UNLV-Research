#!/usr/bin/env python3
"""Build the frozen code-domain confirmatory decision report."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PROTOCOL = Path("configs") / "code_domain_confirmatory_protocol_qwen3_4b_v1.json"
DEFAULT_PROTOCOL_REPORT = OUTPUT_DIR / "validation" / "code_domain_confirmatory_protocol_qwen3_4b_report.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_confirmatory_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_confirmatory_decision_report.json"
DEFAULT_EVALPLUS_REPORT = OUTPUT_DIR / "validation" / "code_domain_evalplus_confirmatory_guardrail_report.json"
DEFAULT_GENERAL_TEXT_REPORT = OUTPUT_DIR / "validation" / "code_domain_general_text_confirmatory_guardrail_report.json"
DEFAULT_GENERAL_TASK_REPORT = OUTPUT_DIR / "validation" / "code_domain_general_task_confirmatory_guardrail_report.json"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "known_high_quality_equal_budget",
)


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


def _paired_delta(left: Dict[int, float], right: Dict[int, float], *, label: str) -> Dict[str, Any]:
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
    protocol_path: Path,
    protocol_report_path: Path,
    output_dir: Path,
    output_path: Path,
    evalplus_report_path: Path,
    general_text_report_path: Path,
    general_task_report_path: Path,
) -> Dict[str, Any]:
    protocol = load_json(protocol_path)
    protocol_report = load_json(protocol_report_path)
    recipe = protocol["confirmatory_training_recipe"]
    steps = int(recipe["optimizer_steps"])
    seeds = [int(seed) for seed in recipe["confirmatory_training_seeds"]]
    margin = float(
        protocol["primary_success_rule"][
            "curated_vs_stageA_random_required_absolute_nll_reduction"
        ]
    )
    blockers: List[str] = []
    source_sha256: Dict[str, str] = {
        str(protocol_path): sha256_file(protocol_path),
        str(protocol_report_path): sha256_file(protocol_report_path),
    }
    evalplus_guardrail = _load_guardrail_report(
        evalplus_report_path,
        "evalplus_confirmatory_guardrail_passed",
    )
    general_text_guardrail = _load_guardrail_report(
        general_text_report_path,
        "general_text_confirmatory_guardrail_passed",
    )
    general_task_guardrail = _load_guardrail_report(
        general_task_report_path,
        "general_task_confirmatory_guardrail_passed",
    )
    for path in (evalplus_report_path, general_text_report_path, general_task_report_path):
        if path.exists():
            source_sha256[str(path)] = sha256_file(path)

    train_results: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for arm in TRAINED_ARMS:
        for seed in seeds:
            path = _run_result_path(output_dir, arm, seed, steps)
            row = _load_completed_json(path, "confirmatory_qlora_completed", blockers)
            if row is None:
                continue
            if int(row.get("optimizer_steps") or -1) != steps:
                blockers.append(f"optimizer_step_mismatch:{arm}:{seed}")
            if int(row.get("seed") or -1) != seed or row.get("arm") != arm:
                blockers.append(f"run_identity_mismatch:{arm}:{seed}")
            train_results[(arm, seed)] = row
            source_sha256[str(path)] = sha256_file(path)

    eval_results: Dict[str, Dict[int, float]] = {arm: {} for arm in TRAINED_ARMS}
    base_result = _load_completed_json(
        _nll_result_path(output_dir, "base_no_update"),
        "heldout_nll_completed",
        blockers,
    )
    if base_result is not None:
        source_sha256[str(_nll_result_path(output_dir, "base_no_update"))] = sha256_file(
            _nll_result_path(output_dir, "base_no_update")
        )
    for arm in TRAINED_ARMS:
        for seed in seeds:
            path = _nll_result_path(output_dir, arm, seed)
            row = _load_completed_json(path, "heldout_nll_completed", blockers)
            if row is None:
                continue
            if row.get("stage") != "confirmatory":
                blockers.append(f"stage_mismatch:{path}:{row.get('stage')}")
            if int(row.get("seed") or -1) != seed or row.get("arm") != arm:
                blockers.append(f"nll_identity_mismatch:{arm}:{seed}")
            eval_results[arm][seed] = float(row["mean_nll"])
            source_sha256[str(path)] = sha256_file(path)

    expected_train = len(TRAINED_ARMS) * len(seeds)
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
    nll_gate: Dict[str, Any]
    if blockers:
        nll_gate = {"status": "incomplete", "blockers": blockers}
        overall_status = "confirmatory_decision_incomplete"
    else:
        primary = protocol["primary_comparison"]
        curated = eval_results[primary["treatment"]]
        stage_a = eval_results[primary["primary_baseline"]]
        raw = eval_results["raw_random_equal_budget"]
        reference = eval_results[primary["reference_arm"]]
        curated_mean = arm_summaries["curated_equal_budget"]["mean_nll"]
        stage_a_mean = arm_summaries["stageA_random_equal_budget"]["mean_nll"]
        raw_mean = arm_summaries["raw_random_equal_budget"]["mean_nll"]
        reference_mean = arm_summaries["known_high_quality_equal_budget"]["mean_nll"]
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
        guardrail_issues = []
        if evalplus_guardrail["status"] == "missing":
            guardrail_issues.append("EvalPlus confirmatory guardrail evidence is missing")
        elif not evalplus_guardrail["passed"]:
            guardrail_issues.append(
                f"EvalPlus confirmatory guardrail is not passing: {evalplus_guardrail['status']}"
            )
        if general_text_guardrail["status"] == "missing":
            guardrail_issues.append("general-text NLL confirmatory retention evidence is missing")
        elif not general_text_guardrail["passed"]:
            guardrail_issues.append(
                "general-text NLL confirmatory retention guardrail is not passing: "
                f"{general_text_guardrail['status']}"
            )
        if general_task_guardrail["status"] == "missing":
            guardrail_issues.append("general-task confirmatory retention evidence is missing")
        elif not general_task_guardrail["passed"]:
            guardrail_issues.append(
                f"general-task confirmatory retention guardrail is not passing: {general_task_guardrail['status']}"
            )
        nll_gate["required_guardrail_issues"] = guardrail_issues
        if not primary_pass:
            overall_status = "confirmatory_decision_reject_primary_margin_failure"
        elif not raw_direction_pass:
            overall_status = "confirmatory_decision_reject_raw_direction_failure"
        elif any(
            guardrail["status"] != "missing" and not guardrail["passed"]
            for guardrail in (evalplus_guardrail, general_text_guardrail, general_task_guardrail)
        ):
            overall_status = "confirmatory_decision_reject_guardrail_failure"
        elif guardrail_issues:
            overall_status = "confirmatory_decision_abstain_missing_required_guardrails"
        else:
            overall_status = "confirmatory_decision_passed"

    report = {
        "schema_version": "code-domain-confirmatory-decision-report-v1",
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
                "evalplus_confirmatory": {
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
                    "passed": general_task_guardrail["passed"],
                    "blocker_count": len(general_task_guardrail["blockers"]),
                    "blockers": general_task_guardrail["blockers"][:10],
                },
            },
            "primary_success_rule": protocol["primary_success_rule"],
            "blockers": blockers,
        },
        "protocol_freeze_summary": protocol_report.get("summary", {}),
        "confirmatory_outcomes_read": completed_eval > 0 or bool(train_results),
        "utility_scope": protocol["utility_scope"],
        "claim_boundary": (
            "Confirmatory decision report. It may read frozen confirmatory outcomes, "
            "but those outcomes must not alter selector objectives, seeds, margins, "
            "token budgets, heldouts, splits, or Stage-C guardrail thresholds."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain confirmatory decision report.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-report", type=Path, default=DEFAULT_PROTOCOL_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evalplus-report", type=Path, default=DEFAULT_EVALPLUS_REPORT)
    parser.add_argument("--general-text-report", type=Path, default=DEFAULT_GENERAL_TEXT_REPORT)
    parser.add_argument("--general-task-report", type=Path, default=DEFAULT_GENERAL_TASK_REPORT)
    args = parser.parse_args()
    report = build(
        args.protocol,
        args.protocol_report,
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
    return 0 if report["status"] != "confirmatory_decision_incomplete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
