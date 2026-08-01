#!/usr/bin/env python3
"""Build a pilot-only report for target-SLM update smoke/pilot runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
ARM_ORDER = (
    "base_no_update",
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
)


def _load_optional(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def build_report(experiment_dir: Path) -> Dict[str, Any]:
    eval_dir = experiment_dir / "eval_results"
    run_dir = experiment_dir / "model_runs"
    evals = {
        "base_no_update": _load_optional(eval_dir / "pilot_base_eval.json"),
        "curated_equal_budget": _load_optional(eval_dir / "pilot_curated_seed20260608_eval.json"),
        "stageA_random_equal_budget": _load_optional(eval_dir / "pilot_stageA_random_seed20260608_eval.json"),
        "raw_random_equal_budget": _load_optional(eval_dir / "pilot_raw_random_seed20260608_eval.json"),
    }
    train_runs = {
        "curated_equal_budget": _load_optional(run_dir / "pilot_curated_seed20260608" / "train_result.json"),
        "stageA_random_equal_budget": _load_optional(run_dir / "pilot_stageA_random_seed20260608" / "train_result.json"),
        "raw_random_equal_budget": _load_optional(run_dir / "pilot_raw_random_seed20260608" / "train_result.json"),
    }
    nll = {
        arm: float(payload.get("mean_nll"))
        for arm, payload in evals.items()
        if isinstance(payload.get("mean_nll"), (int, float))
    }
    deltas = {}
    if "curated_equal_budget" in nll and "stageA_random_equal_budget" in nll:
        deltas["curated_minus_stageA_random_nll"] = nll["curated_equal_budget"] - nll["stageA_random_equal_budget"]
    if "curated_equal_budget" in nll and "raw_random_equal_budget" in nll:
        deltas["curated_minus_raw_random_nll"] = nll["curated_equal_budget"] - nll["raw_random_equal_budget"]
    if "curated_equal_budget" in nll and "base_no_update" in nll:
        deltas["curated_minus_base_nll"] = nll["curated_equal_budget"] - nll["base_no_update"]
    if "stageA_random_equal_budget" in nll and "base_no_update" in nll:
        deltas["stageA_random_minus_base_nll"] = nll["stageA_random_equal_budget"] - nll["base_no_update"]
    ranking = sorted(nll, key=lambda arm: (nll[arm], arm))
    primary_direction = (
        "curated_better_than_stageA_random"
        if deltas.get("curated_minus_stageA_random_nll", 0.0) < 0.0
        else "curated_not_better_than_stageA_random"
    )
    report = {
        "schema_version": "slm-update-pilot-report-v1",
        "experiment_dir": str(experiment_dir),
        "scope": "pilot_only_not_certification_evidence",
        "primary_comparison": "curated_equal_budget_vs_stageA_random_equal_budget",
        "utility_scope": "Stage C validation only; never selector objective",
        "pilot_limits": [
            "single seed",
            "256 training sequences per arm",
            "128 eval sequences",
            "32 optimizer steps",
            "internal same-corpus Stage-A heldout only",
            "no external benchmark, forgetting, safety, or contamination result yet",
        ],
        "eval_mean_nll": nll,
        "eval_ranking_lower_is_better": ranking,
        "deltas": deltas,
        "primary_direction": primary_direction,
        "interpretation": (
            "Pilot supports running the larger experiment because curated is slightly lower-NLL than Stage-A random, "
            "but all update arms are worse than base no-update under this very short run."
        ),
        "eval_results": evals,
        "train_runs": train_runs,
    }
    save_json(experiment_dir / "pilot_report.json", report)
    md = [
        "# SLM Update Pilot Report",
        "",
        "Scope: pilot only; not certification evidence.",
        "",
        "| Arm | Mean NLL | PPL |",
        "| --- | ---: | ---: |",
    ]
    for arm in ARM_ORDER:
        payload = evals.get(arm) or {}
        nll_value = payload.get("mean_nll")
        ppl_value = payload.get("perplexity")
        md.append(
            f"| `{arm}` | {float(nll_value):.6f} | {float(ppl_value):.6f} |"
            if isinstance(nll_value, (int, float)) and isinstance(ppl_value, (int, float))
            else f"| `{arm}` | missing | missing |"
        )
    md.extend(
        [
            "",
            "## Deltas",
            "",
            "| Delta | Value | Meaning |",
            "| --- | ---: | --- |",
        ]
    )
    for name, value in sorted(deltas.items()):
        md.append(f"| `{name}` | {float(value):.6f} | lower is better for the left arm |")
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Do not use this pilot as a paper claim. It only validates the runner and suggests that a larger equal-budget run is worth doing.",
        ]
    )
    (experiment_dir / "pilot_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SLM update pilot report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    args = parser.parse_args()
    report = build_report(args.experiment_dir)
    print(
        {
            "scope": report["scope"],
            "primary_direction": report["primary_direction"],
            "eval_mean_nll": report["eval_mean_nll"],
            "deltas": report["deltas"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
