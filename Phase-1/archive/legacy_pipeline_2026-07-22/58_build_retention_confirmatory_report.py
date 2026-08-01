#!/usr/bin/env python3
"""Build the frozen retention-aware recipe confirmatory report."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path("configs") / "retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json"


def _nll(path: Path) -> float:
    value = load_json(path).get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing mean_nll: {path}")
    return float(value)


def build_report(experiment_dir: Path, plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_dir = experiment_dir / "eval_results"
    base_external = _nll(eval_dir / "retention_confirm_seed20260612_base_external.json")
    rows = []
    for seed in plan["fresh_training_seeds"]:
        stagea_target = _nll(eval_dir / f"retention_confirm_seed{seed}_stageA_target.json")
        candidate_target = _nll(eval_dir / f"retention_confirm_seed{seed}_candidate_target.json")
        candidate_external = _nll(eval_dir / f"retention_confirm_seed{seed}_candidate_external.json")
        target_gain = stagea_target - candidate_target
        external_regression = candidate_external - base_external
        rows.append(
            {
                "seed": seed,
                "stageA_target_nll": stagea_target,
                "candidate_target_nll": candidate_target,
                "target_improvement_vs_stageA": target_gain,
                "base_external_nll": base_external,
                "candidate_external_nll": candidate_external,
                "external_regression_vs_base": external_regression,
                "target_pass": target_gain > 0.0,
                "external_pass": external_regression <= 0.0,
                "joint_pass": target_gain > 0.0 and external_regression <= 0.0,
            }
        )
    overall_pass = all(row["joint_pass"] for row in rows)
    target_gains = [float(row["target_improvement_vs_stageA"]) for row in rows]
    external_regressions = [float(row["external_regression_vs_base"]) for row in rows]
    report = {
        "schema_version": "retention-recipe-confirmatory-report-v1",
        "status": "confirmatory_joint_supported" if overall_pass else "confirmatory_joint_not_supported",
        "plan": plan,
        "rows": rows,
        "summary": {
            "target_improvement_mean": statistics.mean(target_gains),
            "target_improvement_stdev": statistics.stdev(target_gains) if len(target_gains) > 1 else 0.0,
            "target_improvement_min": min(target_gains),
            "external_regression_mean": statistics.mean(external_regressions),
            "external_regression_max": max(external_regressions),
            "joint_pass_seed_count": sum(1 for row in rows if row["joint_pass"]),
            "seed_count": len(rows),
        },
        "overall_pass": overall_pass,
        "interpretation": (
            "The frozen replay-aware recipe consistently preserves the external retention outcome, "
            "but its target advantage over matched Stage-A random is not seed-stable."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Internal target and provisional external-corpus retention confirmation only. "
            "No deployment, task-capability, safety, or semantic-contamination claim."
        ),
    }
    save_json(experiment_dir / "retention_recipe_confirmatory_report.json", report)
    lines = [
        "# Retention Recipe Confirmatory Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Seed | Target gain vs Stage-A | External regression vs base | Joint pass |",
        "| ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['target_improvement_vs_stageA']:.9f} | "
            f"{row['external_regression_vs_base']:.9f} | {row['joint_pass']} |"
        )
    lines.extend(["", "## Interpretation", "", report["interpretation"], "", "## Claim Boundary", "", report["claim_boundary"], ""])
    (experiment_dir / "retention_recipe_confirmatory_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retention recipe confirmatory report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan)
    print({"status": report["status"], "overall_pass": report["overall_pass"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
