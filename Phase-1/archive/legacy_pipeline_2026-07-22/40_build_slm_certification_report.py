#!/usr/bin/env python3
"""Build certification-scale target-SLM report from completed runs."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path(__file__).resolve().parent / "configs" / "slm_update_certification_plan_qwen25_0p5b_fineweb.json"


def _load_optional(path: Path) -> Dict[str, Any]:
    return load_json(path) if path.exists() else {}


def _mean(values: List[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _stdev(values: List[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) >= 2 else None


def _nll(payload: Dict[str, Any]) -> float | None:
    value = payload.get("mean_nll")
    return float(value) if isinstance(value, (int, float)) else None


def build_report(experiment_dir: Path, plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_dir = experiment_dir / "eval_results"
    run_dir = experiment_dir / "model_runs"
    prefix = "cert_lr1e5_full"
    seeds = [int(seed) for seed in plan.get("seeds", [])]
    base_eval = _load_optional(eval_dir / f"{prefix}_base_eval.json")
    rows = []
    deltas = []
    complete_seed_count = 0
    for seed in seeds:
        curated_eval = _load_optional(eval_dir / f"{prefix}_curated_seed{seed}_eval.json")
        stagea_eval = _load_optional(eval_dir / f"{prefix}_stageA_random_seed{seed}_eval.json")
        curated_train = _load_optional(run_dir / f"{prefix}_curated_seed{seed}" / "train_result.json")
        stagea_train = _load_optional(run_dir / f"{prefix}_stageA_random_seed{seed}" / "train_result.json")
        curated_nll = _nll(curated_eval)
        stagea_nll = _nll(stagea_eval)
        delta = None
        complete = curated_nll is not None and stagea_nll is not None
        if complete:
            complete_seed_count += 1
            delta = curated_nll - stagea_nll
            deltas.append(delta)
        rows.append(
            {
                "seed": seed,
                "complete": complete,
                "curated_mean_nll": curated_nll,
                "stageA_random_mean_nll": stagea_nll,
                "curated_minus_stageA_random_nll": delta,
                "curated_train_steps": curated_train.get("steps"),
                "stageA_random_train_steps": stagea_train.get("steps"),
                "curated_train_mean_loss": curated_train.get("mean_loss"),
                "stageA_random_train_mean_loss": stagea_train.get("mean_loss"),
            }
        )
    curated_values = [float(row["curated_mean_nll"]) for row in rows if row["curated_mean_nll"] is not None]
    stagea_values = [float(row["stageA_random_mean_nll"]) for row in rows if row["stageA_random_mean_nll"] is not None]
    base_nll = _nll(base_eval)
    summary = {
        "base_no_update_mean_nll": base_nll,
        "complete_seed_count": complete_seed_count,
        "planned_seed_count": len(seeds),
        "curated_mean_nll_mean": _mean(curated_values),
        "curated_mean_nll_stdev": _stdev(curated_values),
        "stageA_random_mean_nll_mean": _mean(stagea_values),
        "stageA_random_mean_nll_stdev": _stdev(stagea_values),
        "curated_minus_stageA_random_nll_mean": _mean(deltas),
        "curated_minus_stageA_random_nll_stdev": _stdev(deltas),
        "curated_better_seed_count": sum(1 for value in deltas if value < 0.0),
        "stageA_random_better_seed_count": sum(1 for value in deltas if value > 0.0),
    }
    if base_nll is not None and summary["curated_mean_nll_mean"] is not None:
        summary["curated_mean_minus_base_nll"] = float(summary["curated_mean_nll_mean"]) - base_nll
    if base_nll is not None and summary["stageA_random_mean_nll_mean"] is not None:
        summary["stageA_random_mean_minus_base_nll"] = float(summary["stageA_random_mean_nll_mean"]) - base_nll

    status = "incomplete"
    if complete_seed_count >= 3:
        status = (
            "primary_success_candidate"
            if summary["curated_minus_stageA_random_nll_mean"] is not None
            and float(summary["curated_minus_stageA_random_nll_mean"]) < 0.0
            and int(summary["curated_better_seed_count"]) >= 2
            else "primary_not_supported"
        )
    elif complete_seed_count > 0 and int(summary["curated_better_seed_count"]) == 0:
        status = "early_negative_signal_pause_recommended"

    report = {
        "schema_version": "slm-update-certification-report-v1",
        "scope": "certification_scale_internal_heldout",
        "status": status,
        "plan_path": str(plan_path),
        "experiment_dir": str(experiment_dir),
        "primary_comparison": plan.get("primary_comparison"),
        "utility_scope": "Stage C validation only; never selector objective",
        "summary": summary,
        "seed_rows": rows,
        "base_eval": base_eval,
        "interpretation": (
            "Completed certification-scale seeds do not yet support the curated arm. "
            "Pause before spending more GPU time if the first complete seed is negative, "
            "because full runs are expensive and the scaled pilot may not transfer to full-budget training."
            if status == "early_negative_signal_pause_recommended"
            else "Certification-scale report should be interpreted only after all planned primary seeds complete."
        ),
        "claim_boundary": plan.get("claim_boundary"),
    }
    save_json(experiment_dir / f"{prefix}_certification_report.json", report)
    md_lines = [
        "# SLM Update Certification Report",
        "",
        f"Status: `{status}`",
        "",
        "Scope: internal heldout certification-scale run; external benchmark, forgetting, safety, and contamination checks are still separate.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            md_lines.append(f"| `{key}` | {value:.9f} |")
        else:
            md_lines.append(f"| `{key}` | {value} |")
    md_lines.extend(["", "## Seeds", "", "| Seed | Complete | Curated NLL | Stage-A Random NLL | Delta |", "| --- | --- | ---: | ---: | ---: |"])
    for row in rows:
        curated = row["curated_mean_nll"]
        stagea = row["stageA_random_mean_nll"]
        delta = row["curated_minus_stageA_random_nll"]
        curated_cell = f"{curated:.9f}" if isinstance(curated, float) else "missing"
        stagea_cell = f"{stagea:.9f}" if isinstance(stagea, float) else "missing"
        delta_cell = f"{delta:.9f}" if isinstance(delta, float) else "missing"
        md_lines.append(f"| {row['seed']} | {row['complete']} | {curated_cell} | {stagea_cell} | {delta_cell} |")
    md_lines.extend(["", "## Interpretation", "", report["interpretation"], ""])
    (experiment_dir / f"{prefix}_certification_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SLM certification report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan)
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
