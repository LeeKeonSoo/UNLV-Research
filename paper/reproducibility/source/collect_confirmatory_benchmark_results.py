#!/usr/bin/env python3
"""Collect official confirmatory benchmark outputs on one percent scale."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Final


DEFAULT_ROOT: Final = Path(
    "D:/UNLV-Research/final_all_policy_v1/external_training_v1/benchmarks_v1"
)
PRIMARY_REASONING_BENCHMARKS: Final = (
    "BigCodeBench Complete",
    "CRUXEval-I",
    "CRUXEval-O",
    "DS-1000",
)
SECONDARY_SHORT_FUNCTION_BENCHMARKS: Final = (
    "HumanEval+",
    "MBPP+",
)
BENCHMARKS: Final = (
    *PRIMARY_REASONING_BENCHMARKS,
    *SECONDARY_SHORT_FUNCTION_BENCHMARKS,
)
MODEL_ARMS: Final = (
    ("base_no_update", None),
    ("raw_audited_natural", 101),
    ("raw_audited_natural", 202),
    ("raw_audited_natural", 303),
    ("normal_natural", 101),
    ("normal_natural", 202),
    ("normal_natural", 303),
    ("hard_natural", 101),
    ("hard_natural", 202),
    ("hard_natural", 303),
)
TRAINED_ARMS: Final = (
    "raw_audited_natural",
    "normal_natural",
    "hard_natural",
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def parse_evalplus_percent(path: Path) -> float:
    evaluation = _json_object(path).get("eval")
    if not isinstance(evaluation, dict) or not evaluation:
        raise ValueError(f"EvalPlus output has no task results: {path}")
    passed = 0
    total = 0
    for task_results in evaluation.values():
        if not isinstance(task_results, list) or not task_results:
            raise ValueError(f"EvalPlus task has no generations: {path}")
        result = task_results[0]
        if not isinstance(result, dict) or not {
            "base_status",
            "plus_status",
        }.issubset(result):
            raise ValueError(f"EvalPlus task has no base/plus status pair: {path}")
        passed += result["base_status"] == result["plus_status"] == "pass"
        total += 1
    return round(100.0 * passed / total, 6)


def parse_bigcodebench_percent(path: Path) -> float:
    value = _json_object(path).get("pass@1")
    if not isinstance(value, (int, float)):
        raise ValueError(f"BigCodeBench output has no numeric pass@1: {path}")
    score = float(value)
    return round(score * 100.0 if score <= 1.0 else score, 6)


def parse_cruxeval_percent(path: Path) -> float:
    value = _json_object(path).get("pass_at_1")
    if not isinstance(value, (int, float)):
        raise ValueError(f"CRUXEval output has no numeric pass_at_1: {path}")
    score = float(value)
    return round(score * 100.0 if score <= 1.0 else score, 6)


def parse_ds1000_percent(path: Path) -> float:
    text = path.read_text(encoding="utf-8-sig")
    match = re.search(r"^mean\s+([0-9]*\.?[0-9]+)\s*$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"DS-1000 output has no mean score: {path}")
    score = float(match.group(1))
    return round(score * 100.0 if score <= 1.0 else score, 6)


def _suffix(seed: int | None) -> str:
    return "base" if seed is None else f"seed{seed}"


def _model_label(arm: str, seed: int | None) -> str:
    if seed is None:
        return "Base"
    labels = {
        "raw_audited_natural": "Raw",
        "normal_natural": "Normal",
        "hard_natural": "Hard",
    }
    return f"{labels[arm]} seed {seed}"


def result_paths(root: Path, arm: str, seed: int | None) -> dict[str, Path]:
    suffix = _suffix(seed)
    samples = root / "samples"
    results = root / "official_results"
    return {
        "HumanEval+": samples / "evalplus" / f"humaneval_{arm}_{suffix}_eval_results.json",
        "MBPP+": samples / "evalplus" / f"mbpp_{arm}_{suffix}_eval_results.json",
        "BigCodeBench Complete": results / f"bigcodebench_{arm}_{suffix}.json",
        "CRUXEval-I": results / f"cruxeval_input_{arm}_{suffix}.json",
        "CRUXEval-O": results / f"cruxeval_output_{arm}_{suffix}.json",
        "DS-1000": results / f"ds1000_{arm}_{suffix}.txt",
    }


def selected_model_arms(seeds: tuple[int, ...]) -> tuple[tuple[str, int | None], ...]:
    if not seeds:
        raise ValueError("At least one seed is required")
    return (("base_no_update", None),) + tuple(
        (arm, seed) for arm in TRAINED_ARMS for seed in seeds
    )


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for arm in TRAINED_ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if not arm_rows:
            continue
        scores: dict[str, dict[str, float | int]] = {}
        for benchmark in BENCHMARKS:
            values = [float(row["scores_percent"][benchmark]) for row in arm_rows]
            scores[benchmark] = {
                "mean_percent": round(statistics.fmean(values), 6),
                "sample_std_percent": round(statistics.stdev(values), 6)
                if len(values) > 1
                else 0.0,
                "seed_count": len(values),
            }
        primary_macro_values = [
            float(scores[benchmark]["mean_percent"])
            for benchmark in PRIMARY_REASONING_BENCHMARKS
        ]
        summaries.append(
            {
                "arm": arm,
                "scores": scores,
                "primary_reasoning_macro_percent": round(
                    statistics.fmean(primary_macro_values), 6
                ),
            }
        )
    return summaries


def collect(root: Path, seeds: tuple[int, ...] = (101, 202, 303)) -> dict[str, Any]:
    parsers = {
        "HumanEval+": parse_evalplus_percent,
        "MBPP+": parse_evalplus_percent,
        "BigCodeBench Complete": parse_bigcodebench_percent,
        "CRUXEval-I": parse_cruxeval_percent,
        "CRUXEval-O": parse_cruxeval_percent,
        "DS-1000": parse_ds1000_percent,
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm, seed in selected_model_arms(seeds):
        paths = result_paths(root, arm, seed)
        scores: dict[str, float] = {}
        for benchmark, path in paths.items():
            if not path.is_file():
                missing.append(str(path))
                continue
            scores[benchmark] = parsers[benchmark](path)
        rows.append(
            {
                "model_arm": _model_label(arm, seed),
                "arm": arm,
                "seed": seed,
                "scores_percent": scores,
                "primary_reasoning_macro_percent": round(
                    statistics.fmean(
                        scores[benchmark]
                        for benchmark in PRIMARY_REASONING_BENCHMARKS
                    ),
                    6,
                ),
            }
        )
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} official result files:\n{preview}")
    return {
        "schema_version": "confirmatory-benchmark-results-v1",
        "score_unit": "percent",
        "seeds": list(seeds),
        "evidence_scope": "exploratory_two_seed" if len(seeds) == 2 else "confirmatory_three_seed",
        "benchmarks": list(BENCHMARKS),
        "benchmark_hierarchy": {
            "amendment": "protocols/code_reasoning_primary_amendment_v1.json",
            "primary_reasoning": list(PRIMARY_REASONING_BENCHMARKS),
            "secondary_short_function": list(SECONDARY_SHORT_FUNCTION_BENCHMARKS),
            "all_results_mandatory": True,
        },
        "rows": rows,
        "arm_summaries": summarize_rows(rows),
    }


def markdown_table(report: dict[str, Any]) -> str:
    benchmarks = list(report["benchmarks"])
    lines = [
        "Primary outcome: reasoning-suite macro. HumanEval+ and MBPP+ are mandatory secondary diagnostics.",
        "",
        "| Model arm | Reasoning macro | " + " | ".join(benchmarks) + " |",
        "|---|---:|" + "---:|" * len(benchmarks),
    ]
    for row in report["rows"]:
        scores = row["scores_percent"]
        values = [f"{float(scores[name]):.2f}%" for name in benchmarks]
        lines.append(
            f"| {row['model_arm']} | "
            f"{float(row['primary_reasoning_macro_percent']):.2f}% | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Across-seed summary (mean +/- sample standard deviation):",
            "",
            "| Arm | Reasoning macro | " + " | ".join(benchmarks) + " |",
            "|---|---:|" + "---:|" * len(benchmarks),
        ]
    )
    labels = {
        "raw_audited_natural": "Raw",
        "normal_natural": "Normal",
        "hard_natural": "Hard",
    }
    for summary in report["arm_summaries"]:
        values = []
        for benchmark in benchmarks:
            score = summary["scores"][benchmark]
            values.append(
                f"{float(score['mean_percent']):.2f}% +/- "
                f"{float(score['sample_std_percent']):.2f}"
            )
        lines.append(
            f"| {labels[summary['arm']]} | "
            f"{float(summary['primary_reasoning_macro_percent']):.2f}% | "
            + " | ".join(values)
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()
    report = collect(args.root, tuple(args.seeds))
    json_path = args.root / "confirmatory_benchmark_results.json"
    markdown_path = args.root / "confirmatory_benchmark_results.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_table(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
