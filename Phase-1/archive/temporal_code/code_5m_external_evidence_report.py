#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
DATASETS = {"humaneval": "HumanEval+", "mbpp": "MBPP+"}
BASE_ARM = "base_no_update"
RAW_ARM = "raw_safe_natural"
CURATED_ARM = "curated_natural"
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPLICATES = 10_000
DEFAULT_PLAN = Path("configs") / "code_5m_natural_budget_execution_qwen3_4b_v1.json"
DEFAULT_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_RUN_DIR = Path(r"D:\UNLV-Research\code_5m_corpus_v2\external_validation_v1\runs")
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_5m_natural_budget_external_evidence_report.json"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _seeds(plan: JsonMap) -> list[int]:
    return [int(seed) for seed in plan["training_recipe"]["development_training_seeds"]]


def _input_report_path(plan: JsonMap) -> Path:
    return Path(str(plan["frozen_inputs"]["materialization_report"]))


def _result_path(run_dir: Path, dataset: str, arm: str, seed: int | None) -> Path:
    suffix = "base" if seed is None else f"seed{seed}"
    return run_dir / "evalplus_guardrail" / "results" / f"{dataset}_{arm}_{suffix}_eval.json"


def _nll_path(run_dir: Path, arm: str, seed: int | None) -> Path:
    suffix = "base_no_update" if seed is None else f"{arm}_seed{seed}"
    return run_dir / "heldout_nll" / f"{suffix}.json"


def _paths(plan_path: Path, split_path: Path, plan: JsonMap, run_dir: Path) -> list[Path]:
    seeds = _seeds(plan)
    paths = [plan_path, split_path, _input_report_path(plan), _nll_path(run_dir, BASE_ARM, None)]
    for arm in (RAW_ARM, CURATED_ARM):
        paths.extend(_nll_path(run_dir, arm, seed) for seed in seeds)
        for dataset in DATASETS:
            paths.extend(_result_path(run_dir, dataset, arm, seed) for seed in seeds)
    paths.extend(_result_path(run_dir, dataset, BASE_ARM, None) for dataset in DATASETS)
    return paths


def _expected_task_counts(split: JsonMap) -> dict[str, int]:
    counts = split["summary"]["suite_split_counts"]
    return {dataset: int(counts[f"{label}/development"]) for dataset, label in DATASETS.items()}


def _task_count_blockers(run_dir: Path, plan: JsonMap, expected_counts: dict[str, int]) -> list[str]:
    blockers: list[str] = []
    for dataset, expected_count in expected_counts.items():
        for arm, seeds in ((BASE_ARM, [None]), (RAW_ARM, _seeds(plan)), (CURATED_ARM, _seeds(plan))):
            for seed in seeds:
                path = _result_path(run_dir, dataset, arm, seed)
                row = load_json(path)
                if row.get("status") != "evalplus_samples_evaluated":
                    blockers.append(f"result_status_mismatch:{path}")
                if int(row["task_count"]) != expected_count:
                    blockers.append(f"task_count_mismatch:{path}:{row['task_count']}!={expected_count}")
    return blockers


def _bootstrap_interval(deltas: list[float]) -> JsonMap:
    generator = random.Random(BOOTSTRAP_SEED)
    estimates = [_mean([generator.choice(deltas) for _ in deltas]) for _ in range(BOOTSTRAP_REPLICATES)]
    return {
        "method": "paired_seed_nonparametric_bootstrap",
        "fixed_seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "interval_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "limitation": "The interval is descriptive with only 3 training seeds and is not a significance claim.",
    }


def _evalplus_arm(run_dir: Path, arm: str, seeds: list[int] | None) -> JsonMap:
    suites: JsonMap = {}
    seed_macros: dict[str, list[float]] = {str(seed): [] for seed in seeds or []}
    for dataset, label in DATASETS.items():
        if seeds is None:
            row = load_json(_result_path(run_dir, dataset, arm, None))
            suites[label] = {"pass_rate": float(row["pass_rate"]), "task_count": int(row["task_count"])}
            continue
        rates = {str(seed): float(load_json(_result_path(run_dir, dataset, arm, seed))["pass_rate"]) for seed in seeds}
        for seed, rate in rates.items():
            seed_macros[seed].append(rate)
        suites[label] = {"mean_pass_rate": _mean(list(rates.values())), "per_seed_pass_rate": rates}
    if seeds is None:
        macro = _mean([float(row["pass_rate"]) for row in suites.values()])
        return {"suites": suites, "macro_pass_rate": macro}
    per_seed = {seed: _mean(values) for seed, values in seed_macros.items()}
    return {"suites": suites, "macro_pass_rate": _mean(list(per_seed.values())), "per_seed_macro_pass_rate": per_seed}


def _nll_arm(run_dir: Path, arm: str, seeds: list[int] | None) -> JsonMap:
    if seeds is None:
        row = load_json(_nll_path(run_dir, arm, None))
        return {"mean_nll": float(row["mean_nll"]), "eval_tokens": int(row["tokens"])}
    rates = {str(seed): float(load_json(_nll_path(run_dir, arm, seed))["mean_nll"]) for seed in seeds}
    return {"mean_nll": _mean(list(rates.values())), "per_seed_mean_nll": rates}


def _base_deltas(arms: JsonMap, arm: str) -> JsonMap:
    base_suites = arms[BASE_ARM]["evalplus"]["suites"]
    arm_suites = arms[arm]["evalplus"]["suites"]
    return {
        label: arm_suites[label]["mean_pass_rate"] - base_suites[label]["pass_rate"]
        for label in DATASETS.values()
    } | {
        "macro": arms[arm]["evalplus"]["macro_pass_rate"] - arms[BASE_ARM]["evalplus"]["macro_pass_rate"]
    }


def _markdown(report: JsonMap) -> str:
    lines = ["# code_5m Natural-Budget External Evidence", "", f"Status: `{report['status']}`", "", "## Source SHA256 Freeze Manifest", ""]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in report["source_sha256"].items())
    lines.extend(["", "## Token, Steps, and NLL", "", "| Arm | Training tokens | Optimizer steps | Mean NLL |", "| --- | ---: | ---: | ---: |"])
    for arm, row in report["arms"].items():
        lines.append(f"| {arm} | {row.get('training_tokens', 'n/a')} | {row.get('optimizer_steps', 'n/a')} | {row['nll']['mean_nll']:.6f} |")
    lines.extend(["", "## EvalPlus", "", "| Arm | HumanEval+ | MBPP+ | Macro |", "| --- | ---: | ---: | ---: |"])
    for arm, row in report["arms"].items():
        suites = row["evalplus"]["suites"]
        human = suites["HumanEval+"].get("pass_rate", suites["HumanEval+"].get("mean_pass_rate"))
        mbpp = suites["MBPP+"].get("pass_rate", suites["MBPP+"].get("mean_pass_rate"))
        lines.append(f"| {arm} | {human:.6f} | {mbpp:.6f} | {row['evalplus']['macro_pass_rate']:.6f} |")
    delta = report["raw_vs_curated"]
    base_retention = report["base_retention"]
    raw_minus_base = base_retention["raw_minus_base"]
    curated_minus_base = base_retention["curated_minus_base"]
    lines.extend(["", "## Raw vs Curated", "", f"Seed-level macro deltas (Curated minus Raw): `{delta['per_seed_macro_deltas']}`.", f"Mean delta: `{delta['macro_delta_curated_minus_raw']:.6f}`. Deterministic 95% bootstrap interval: `[{delta['bootstrap_interval']['lower']:.6f}, {delta['bootstrap_interval']['upper']:.6f}]`.", "", "## Base-retention interpretation", "", f"Raw-minus-Base: HumanEval+ `{raw_minus_base['HumanEval+']:.6f}`; MBPP+ `{raw_minus_base['MBPP+']:.6f}`; macro `{raw_minus_base['macro']:.6f}`.", f"Curated-minus-Base: HumanEval+ `{curated_minus_base['HumanEval+']:.6f}`; MBPP+ `{curated_minus_base['MBPP+']:.6f}`; macro `{curated_minus_base['macro']:.6f}`.", f"MBPP+ Base-to-Curated change: `{curated_minus_base['MBPP+']:.6f}`.", base_retention["uncertainty_note"], "", "## Claim Boundary", "", report["claim_boundary"]])
    return "\n".join(lines) + "\n"


def build(plan_path: Path, split_path: Path, run_dir: Path, output_path: Path, markdown_path: Path | None = None) -> JsonMap:
    plan = load_json(plan_path)
    seeds = _seeds(plan)
    source_paths = _paths(plan_path, split_path, plan, run_dir)
    blockers = [f"missing_source:{path}" for path in source_paths if not path.exists()]
    if blockers:
        report = {"schema_version": "code-5m-natural-budget-external-evidence-v1", "status": "external_evidence_incomplete", "blockers": blockers, "source_sha256": {}}
        save_json(output_path, report)
        return report
    split = load_json(split_path)
    expected_task_counts = _expected_task_counts(split)
    blockers = _task_count_blockers(run_dir, plan, expected_task_counts)
    if blockers:
        report = {"schema_version": "code-5m-natural-budget-external-evidence-v1", "status": "external_evidence_incomplete", "expected_suite_task_counts": expected_task_counts, "blockers": blockers, "source_sha256": {str(path): sha256_file(path) for path in source_paths}}
        save_json(output_path, report)
        return report
    input_report = load_json(_input_report_path(plan))
    arms = {
        BASE_ARM: {"training_tokens": 0, "optimizer_steps": 0, "nll": _nll_arm(run_dir, BASE_ARM, None), "evalplus": _evalplus_arm(run_dir, BASE_ARM, None)},
        RAW_ARM: {"training_tokens": int(input_report["arms"][RAW_ARM]["effective_training_tokens"] if "effective_training_tokens" in input_report["arms"][RAW_ARM] else input_report["arms"][RAW_ARM]["packed_tokens"]), "optimizer_steps": int(plan["training_recipe"]["optimizer_steps_by_arm"][RAW_ARM]), "nll": _nll_arm(run_dir, RAW_ARM, seeds), "evalplus": _evalplus_arm(run_dir, RAW_ARM, seeds)},
        CURATED_ARM: {"training_tokens": int(input_report["arms"][CURATED_ARM]["effective_training_tokens"] if "effective_training_tokens" in input_report["arms"][CURATED_ARM] else input_report["arms"][CURATED_ARM]["packed_tokens"]), "optimizer_steps": int(plan["training_recipe"]["optimizer_steps_by_arm"][CURATED_ARM]), "nll": _nll_arm(run_dir, CURATED_ARM, seeds), "evalplus": _evalplus_arm(run_dir, CURATED_ARM, seeds)},
    }
    deltas = {seed: arms[CURATED_ARM]["evalplus"]["per_seed_macro_pass_rate"][seed] - arms[RAW_ARM]["evalplus"]["per_seed_macro_pass_rate"][seed] for seed in map(str, seeds)}
    report = {
        "schema_version": "code-5m-natural-budget-external-evidence-v1",
        "status": "external_evidence_complete",
        "seed_scope": seeds,
        "source_sha256": {str(path): sha256_file(path) for path in source_paths}, "expected_suite_task_counts": expected_task_counts,
        "arms": arms,
        "raw_vs_curated": {"per_seed_macro_deltas": deltas, "macro_delta_curated_minus_raw": _mean(list(deltas.values())), "bootstrap_interval": _bootstrap_interval(list(deltas.values()))},
        "base_retention": {"base_macro_pass_rate": arms[BASE_ARM]["evalplus"]["macro_pass_rate"], "raw_minus_base": _base_deltas(arms, RAW_ARM), "curated_minus_base": _base_deltas(arms, CURATED_ARM), "uncertainty_note": "Base has one deterministic result; no Base uncertainty or significance claim is made."},
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "Descriptive external validation only. It forbids universal, production, intrinsic-quality, and all-suite-improvement claims.",
        "blockers": [],
    }
    save_json(output_path, report)
    effective_markdown_path = markdown_path or output_path.with_suffix(".md")
    effective_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    effective_markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the code_5m natural-budget external-evidence report.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = build(args.plan, args.split, args.run_dir, args.output, args.markdown)
    print(json.dumps({"status": report["status"], "output": str(args.output), "blockers": report["blockers"]}, sort_keys=True))
    return 0 if report["status"] == "external_evidence_complete" else 2
