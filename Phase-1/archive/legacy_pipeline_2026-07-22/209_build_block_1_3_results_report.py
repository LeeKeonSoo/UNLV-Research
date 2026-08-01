from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Final


ROOT: Final = Path(__file__).resolve().parent
OUT: Final = ROOT / "outputs" / "validation"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Block 1-3 Results",
        "",
        "## Block 1: Math selector v2 Stage-C retest",
        "",
        "| Arm | Records | Token proxy | Packed train tokens | Steps | Mean NLL | Std | Delta vs raw | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["block_1_math_v2"]["rows"]:
        lines.append(
            f"| {row['arm']} | {row['records']} | {row['token_proxy_count']} | "
            f"{row['packed_training_tokens']} | {row['optimizer_steps']} | "
            f"{row['mean_nll']:.6f} | {row['sample_std_nll']:.6f} | "
            f"{row['delta_vs_raw']:.6f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Block 2: Integrated Code and Math table",
            "",
            "| Domain/run | Raw tokens | Curated tokens | Token reduction | Raw score | Curated score | Delta | Result |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["block_2_integrated_table"]:
        lines.append(
            f"| {row['domain_run']} | {row['raw_packed_training_tokens']} | "
            f"{row['curated_packed_training_tokens']} | {row['packed_token_reduction_fraction']:.3f} | "
            f"{row['raw_primary_score']:.6f} | {row['curated_primary_score']:.6f} | "
            f"{row['curated_minus_raw']:.6f} | {row['result']} |"
        )
    lines.extend(["", "## Block 3: Analysis", ""])
    for item in report["block_3_analysis"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def nll_stats(output_dir: Path, arm: str) -> dict[str, Any]:
    values: dict[str, float] = {}
    for path in sorted((output_dir / "heldout_nll").glob(f"{arm}_seed*.json")):
        payload = load_json(path)
        values[str(payload["seed"])] = float(payload["mean_nll"])
    scores = list(values.values())
    return {
        "mean_nll": mean(scores),
        "sample_std_nll": stdev(scores) if len(scores) > 1 else math.nan,
        "per_seed_mean_nll": values,
    }


def training_result(output_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    paths = list((output_dir / "qlora_runs").glob(f"{arm}_seed{seed}_steps*/run_result.json"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one run_result for {arm} seed {seed}, found {len(paths)}")
    return load_json(paths[0])


def build_report() -> dict[str, Any]:
    math_dir = ROOT / "outputs" / "math_domain_natural_budget_v2_qwen3_4b"
    freeze = load_json(OUT / "math_domain_natural_budget_v2_freeze_report.json")
    steps = load_json(math_dir / "token_blocks" / "natural_budget_steps_report.json")
    code = load_json(OUT / "code_domain_natural_budget_stage_c_summary_report.json")
    code_evidence = load_json(OUT / "code_paper_evidence_report.json")
    pilot = load_json(OUT / "natural_budget_stage_c_pilot_report.json")

    raw_stats = nll_stats(math_dir, "raw_full_natural")
    curated_stats = nll_stats(math_dir, "curated_math_v2_natural")
    raw_mean = float(raw_stats["mean_nll"])
    curated_mean = float(curated_stats["mean_nll"])
    improvement = raw_mean - curated_mean
    threshold = 0.003
    math_decision = (
        "math_v2_failed_curated_worse_than_raw"
        if improvement < threshold
        else "math_v2_passed_curated_better_than_raw"
    )

    raw_train = training_result(math_dir, "raw_full_natural", 101)
    curated_train = training_result(math_dir, "curated_math_v2_natural", 101)
    rows = [
        {
            "arm": "raw_full_natural",
            "records": freeze["arms"]["raw_full_natural"]["records"],
            "token_proxy_count": freeze["arms"]["raw_full_natural"]["token_proxy_count"],
            "packed_training_tokens": steps["packed_tokens_by_arm"]["raw_full_natural"],
            "optimizer_steps": raw_train["optimizer_steps"],
            "mean_nll": raw_mean,
            "sample_std_nll": raw_stats["sample_std_nll"],
            "delta_vs_raw": 0.0,
            "decision": "reference_raw_full",
        },
        {
            "arm": "curated_math_v2_natural",
            "records": freeze["arms"]["curated_math_v2_natural"]["records"],
            "token_proxy_count": freeze["arms"]["curated_math_v2_natural"]["token_proxy_count"],
            "packed_training_tokens": steps["packed_tokens_by_arm"]["curated_math_v2_natural"],
            "optimizer_steps": curated_train["optimizer_steps"],
            "mean_nll": curated_mean,
            "sample_std_nll": curated_stats["sample_std_nll"],
            "delta_vs_raw": curated_mean - raw_mean,
            "decision": math_decision,
        },
    ]

    math_v1 = pilot["domains"]["math"]
    code_raw = code["arms"]["raw_full_natural"]
    code_curated = code["arms"]["curated_v2_natural"]
    integrated = [
        {
            "domain_run": "Code v2 natural-budget 3-seed NLL",
            "raw_packed_training_tokens": code_raw["packed_training_tokens"],
            "curated_packed_training_tokens": code_curated["packed_training_tokens"],
            "packed_token_reduction_fraction": code["natural_budget_reduction_curated_vs_raw"][
                "packed_training_token_reduction_fraction"
            ],
            "raw_primary_score": code_raw["mean_nll"],
            "curated_primary_score": code_curated["mean_nll"],
            "curated_minus_raw": code["deltas_curated_minus_raw"]["mean_nll_lower_is_better"],
            "result": code_evidence["paper_table_row"]["decision"],
        },
        {
            "domain_run": "Math v1 natural-budget seed101 NLL",
            "raw_packed_training_tokens": math_v1["arms"]["raw_full_natural"]["packed_training_tokens"],
            "curated_packed_training_tokens": math_v1["arms"]["curated_math_natural"][
                "packed_training_tokens"
            ],
            "packed_token_reduction_fraction": math_v1["natural_budget_reduction"][
                "packed_training_token_reduction_fraction"
            ],
            "raw_primary_score": math_v1["arms"]["raw_full_natural"]["mean_nll"],
            "curated_primary_score": math_v1["arms"]["curated_math_natural"]["mean_nll"],
            "curated_minus_raw": math_v1["nll_deltas_lower_is_better"]["curated_minus_raw_full"],
            "result": "fail",
        },
        {
            "domain_run": "Math v2 natural-budget 3-seed NLL",
            "raw_packed_training_tokens": steps["packed_tokens_by_arm"]["raw_full_natural"],
            "curated_packed_training_tokens": steps["packed_tokens_by_arm"]["curated_math_v2_natural"],
            "packed_token_reduction_fraction": 1
            - steps["packed_tokens_by_arm"]["curated_math_v2_natural"]
            / steps["packed_tokens_by_arm"]["raw_full_natural"],
            "raw_primary_score": raw_mean,
            "curated_primary_score": curated_mean,
            "curated_minus_raw": curated_mean - raw_mean,
            "result": "fail" if math_decision.startswith("math_v2_failed") else "pass",
        },
    ]

    return {
        "schema_version": "block-1-3-results-report-v1",
        "status": "block_1_3_results_completed",
        "block_1_math_v2": {
            "decision": math_decision,
            "required_absolute_nll_reduction": threshold,
            "raw_minus_curated_nll": improvement,
            "rows": rows,
            "per_seed": {
                "raw_full_natural": raw_stats["per_seed_mean_nll"],
                "curated_math_v2_natural": curated_stats["per_seed_mean_nll"],
            },
        },
        "block_2_integrated_table": integrated,
        "block_3_analysis": [
            "Historical Code evidence is positive under natural-budget validation, but the current framework requires a Stage-C rerun before confirmatory use.",
            "Math v1 failure was not just seed noise: the v2 selector also fails under 3-seed Stage-C NLL, with curated NLL worse than raw full.",
            "The math failure suggests the current Math Core/Metric/Policy keeps too little useful mass or removes context needed for mathematical language modeling.",
            "This is not Utility leakage: all reported Utility/NLL/EvalPlus measurements are Stage C validation only and are not selector objectives.",
            "Paper claim should be asymmetric for now: the framework has a defensible code-domain success case, but math remains an open failure case requiring selector/core redesign before a broad LM-curation claim.",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Block 1-3 execution report; not a production release claim.",
    }


def main() -> None:
    report = build_report()
    save_json(OUT / "block_1_3_results_report.json", report)
    save_markdown(OUT / "block_1_3_results_report.md", report)
    print(json.dumps({"status": report["status"], "decision": report["block_1_math_v2"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
