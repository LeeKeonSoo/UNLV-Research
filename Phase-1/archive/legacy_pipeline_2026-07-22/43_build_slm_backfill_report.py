#!/usr/bin/env python3
"""Build the exploratory full-budget coverage-backfill comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"


def _nll(payload: Dict[str, Any]) -> float:
    value = payload.get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError("Evaluation result is missing numeric mean_nll.")
    return float(value)


def build_report(experiment_dir: Path) -> Dict[str, Any]:
    eval_dir = experiment_dir / "eval_results"
    run_dir = experiment_dir / "model_runs"
    paths = {
        "base": eval_dir / "cert_lr1e5_full_base_eval.json",
        "selected_only_curated": eval_dir / "cert_lr1e5_full_curated_seed20260608_eval.json",
        "stageA_random": eval_dir / "cert_lr1e5_full_stageA_random_seed20260608_eval.json",
        "coverage_backfilled_interleaved50": eval_dir / "explore_full_backfilled_interleaved50_seed20260608_eval.json",
    }
    evals = {name: load_json(path) for name, path in paths.items()}
    nlls = {name: _nll(payload) for name, payload in evals.items()}
    backfilled = nlls["coverage_backfilled_interleaved50"]
    comparisons = {
        "backfilled_minus_stageA_random_nll": backfilled - nlls["stageA_random"],
        "backfilled_minus_base_nll": backfilled - nlls["base"],
        "backfilled_minus_selected_only_curated_nll": backfilled - nlls["selected_only_curated"],
    }
    report = {
        "schema_version": "slm-backfilled-full-report-v1",
        "scope": "exploratory_release_training_construction_followup",
        "status": (
            "exploratory_direction_supported_single_seed"
            if all(value < 0.0 for value in comparisons.values())
            else "exploratory_direction_not_supported"
        ),
        "seed": 20260608,
        "experiment_dir": str(experiment_dir),
        "arm_manifest": load_json(experiment_dir / "coverage_backfilled_interleaved50_equal_budget_manifest.json"),
        "train_result": load_json(
            run_dir / "explore_full_backfilled_interleaved50_seed20260608" / "train_result.json"
        ),
        "mean_nll": nlls,
        "comparisons": comparisons,
        "utility_scope": "Stage C validation only; never selector objective",
        "interpretation": (
            "The exploratory coverage-backfilled arm is the best current full-budget "
            "condition on the internal Stage-A heldout. This supports a release/training-"
            "construction mixture direction rather than a Utility-optimized Stage-B selector."
        ),
        "claim_boundary": (
            "Post-hoc one-seed exploratory evidence only. Freeze and replicate the mixture "
            "on untouched evaluation evidence before certification or deployment claims."
        ),
    }
    output_json = experiment_dir / "explore_full_backfilled_interleaved50_report.json"
    output_md = experiment_dir / "explore_full_backfilled_interleaved50_report.md"
    save_json(output_json, report)
    rows = [
        ("base_no_update", nlls["base"]),
        ("selected_only_curated", nlls["selected_only_curated"]),
        ("Stage-A_random", nlls["stageA_random"]),
        ("coverage_backfilled_interleaved50", nlls["coverage_backfilled_interleaved50"]),
    ]
    md = [
        "# Exploratory Full-Budget Coverage-Backfill Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Mean NLL",
        "",
        "| Condition | Mean NLL |",
        "| --- | ---: |",
    ]
    md.extend(f"| `{name}` | {value:.9f} |" for name, value in rows)
    md.extend(["", "## Comparisons", "", "| Comparison | Delta NLL |", "| --- | ---: |"])
    md.extend(f"| `{name}` | {value:.9f} |" for name, value in comparisons.items())
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    output_md.write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exploratory coverage-backfill report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args()
    report = build_report(args.experiment_dir)
    print({"status": report["status"], "comparisons": report["comparisons"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
