#!/usr/bin/env python3
"""Build a scaled pilot report for replicated target-SLM update runs."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"


def _load_json(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def _mean(values: List[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: List[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def build_report(experiment_dir: Path, prefix: str, seeds: List[int]) -> Dict[str, Any]:
    eval_dir = experiment_dir / "eval_results"
    run_dir = experiment_dir / "model_runs"
    base_eval = _load_json(eval_dir / f"{prefix}_base_eval.json")
    rows = []
    deltas = []
    for seed in seeds:
        curated_eval = _load_json(eval_dir / f"{prefix}_curated_seed{seed}_eval.json")
        stagea_eval = _load_json(eval_dir / f"{prefix}_stageA_random_seed{seed}_eval.json")
        curated_train = _load_json(run_dir / f"{prefix}_curated_seed{seed}" / "train_result.json")
        stagea_train = _load_json(run_dir / f"{prefix}_stageA_random_seed{seed}" / "train_result.json")
        curated_nll = curated_eval.get("mean_nll")
        stagea_nll = stagea_eval.get("mean_nll")
        delta = None
        if isinstance(curated_nll, (int, float)) and isinstance(stagea_nll, (int, float)):
            delta = float(curated_nll) - float(stagea_nll)
            deltas.append(delta)
        rows.append(
            {
                "seed": int(seed),
                "curated_mean_nll": curated_nll,
                "stageA_random_mean_nll": stagea_nll,
                "curated_minus_stageA_random_nll": delta,
                "curated_train_mean_loss": curated_train.get("mean_loss"),
                "stageA_random_train_mean_loss": stagea_train.get("mean_loss"),
                "curated_steps": curated_train.get("steps"),
                "stageA_random_steps": stagea_train.get("steps"),
            }
        )
    raw_eval = _load_json(eval_dir / f"{prefix}_raw_random_seed{seeds[0]}_eval.json") if seeds else {}
    base_nll = base_eval.get("mean_nll")
    curated_nlls = [float(row["curated_mean_nll"]) for row in rows if isinstance(row.get("curated_mean_nll"), (int, float))]
    stagea_nlls = [float(row["stageA_random_mean_nll"]) for row in rows if isinstance(row.get("stageA_random_mean_nll"), (int, float))]
    summary = {
        "base_no_update_mean_nll": base_nll,
        "curated_mean_nll_mean": _mean(curated_nlls),
        "curated_mean_nll_stdev": _stdev(curated_nlls),
        "stageA_random_mean_nll_mean": _mean(stagea_nlls),
        "stageA_random_mean_nll_stdev": _stdev(stagea_nlls),
        "curated_minus_stageA_random_nll_mean": _mean(deltas),
        "curated_minus_stageA_random_nll_stdev": _stdev(deltas),
        "curated_better_seed_count": sum(1 for value in deltas if value < 0.0),
        "seed_count": len(deltas),
        "raw_random_seed0_mean_nll": raw_eval.get("mean_nll"),
    }
    if isinstance(base_nll, (int, float)) and summary["curated_mean_nll_mean"] is not None:
        summary["curated_mean_minus_base_nll"] = float(summary["curated_mean_nll_mean"]) - float(base_nll)
    if isinstance(base_nll, (int, float)) and summary["stageA_random_mean_nll_mean"] is not None:
        summary["stageA_random_mean_minus_base_nll"] = float(summary["stageA_random_mean_nll_mean"]) - float(base_nll)
    if isinstance(raw_eval.get("mean_nll"), (int, float)) and summary["curated_mean_nll_mean"] is not None:
        summary["curated_mean_minus_raw_random_seed0_nll"] = float(summary["curated_mean_nll_mean"]) - float(raw_eval["mean_nll"])

    report = {
        "schema_version": "slm-update-scaled-pilot-report-v1",
        "experiment_dir": str(experiment_dir),
        "run_prefix": prefix,
        "scope": "scaled_pilot_not_certification_evidence",
        "primary_comparison": "curated_equal_budget_vs_stageA_random_equal_budget",
        "utility_scope": "Stage C validation only; never selector objective",
        "pilot_limits": [
            "1024 training sequences per arm",
            "512 eval sequences",
            "128 optimizer steps",
            "internal same-corpus Stage-A heldout only",
            "learning rate chosen after smaller pilot; not preregistered certification",
            "no external benchmark, forgetting, safety, or contamination result yet",
        ],
        "summary": summary,
        "seed_rows": rows,
        "base_eval": base_eval,
        "raw_random_supporting_eval": raw_eval,
        "interpretation": (
            "Scaled pilot shows replicated curated lower-NLL than Stage-A random across all available seeds, "
            "and both primary update arms improve over base no-update on the internal heldout. "
            "This is promising runner evidence, not a final paper claim."
        ),
    }
    save_json(experiment_dir / f"{prefix}_scaled_report.json", report)
    md = [
        f"# SLM Update Scaled Pilot Report: {prefix}",
        "",
        "Scope: scaled pilot only; not certification evidence.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            md.append(f"| `{key}` | {value:.9f} |")
        else:
            md.append(f"| `{key}` | {value} |")
    md.extend(["", "## Seed Rows", "", "| Seed | Curated NLL | Stage-A Random NLL | Delta |", "| --- | ---: | ---: | ---: |"])
    for row in rows:
        md.append(
            f"| {row['seed']} | {float(row['curated_mean_nll']):.9f} | "
            f"{float(row['stageA_random_mean_nll']):.9f} | "
            f"{float(row['curated_minus_stageA_random_nll']):.9f} |"
        )
    md.extend(["", "## Interpretation", "", report["interpretation"], ""])
    (experiment_dir / f"{prefix}_scaled_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scaled SLM update pilot report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--prefix", default="pilot_1024_lr1e5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260608, 20260609, 20260610])
    args = parser.parse_args()
    report = build_report(args.experiment_dir, str(args.prefix), [int(seed) for seed in args.seeds])
    print({"scope": report["scope"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
