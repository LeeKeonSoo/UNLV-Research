#!/usr/bin/env python3
"""Build the target-gain versus external-retention development Pareto report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"


def _arm_name(target_fraction: float, overrides: Dict[str, str] | None = None) -> str:
    override = (overrides or {}).get(str(target_fraction))
    if override:
        return str(override)
    percent = target_fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"retention_replay_target{int(round(percent)):03d}"
    basis_points = int(round(target_fraction * 10000.0))
    return f"retention_replay_target{basis_points:05d}"


def _nll(path: Path) -> float:
    value = load_json(path).get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing mean_nll: {path}")
    return float(value)


def build_report(experiment_dir: Path, plan_path: Path, output_stem: str) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_dir = experiment_dir / "eval_results"
    target_eval = "confirmatory_coverage_stratified_stageA_eval"
    base_target = _nll(eval_dir / f"context_seed20260608_base_{target_eval}.json")
    stagea_target = _nll(eval_dir / f"pareto_stageA_random_{target_eval}.json")
    base_external = _nll(eval_dir / "guardrail_base_wikitext103.json")
    stagea_external = _nll(eval_dir / "pareto_stageA_random_wikitext103.json")
    rows = []
    for fraction in plan["candidate_target_fractions"]:
        arm = _arm_name(float(fraction), plan.get("arm_name_overrides"))
        target_nll = _nll(eval_dir / f"pareto_{arm}_{target_eval}.json")
        external_nll = _nll(eval_dir / f"pareto_{arm}_wikitext103.json")
        rows.append(
            {
                "arm": arm,
                "target_fraction": float(fraction),
                "replay_fraction": 1.0 - float(fraction),
                "target_mean_nll": target_nll,
                "target_improvement_vs_stageA": stagea_target - target_nll,
                "target_improvement_vs_base": base_target - target_nll,
                "external_mean_nll": external_nll,
                "external_regression_vs_base": external_nll - base_external,
                "external_improvement_vs_stageA": stagea_external - external_nll,
            }
        )
    non_dominated = []
    for row in rows:
        dominated = any(
            other["target_mean_nll"] <= row["target_mean_nll"]
            and other["external_mean_nll"] <= row["external_mean_nll"]
            and (
                other["target_mean_nll"] < row["target_mean_nll"]
                or other["external_mean_nll"] < row["external_mean_nll"]
            )
            for other in rows
        )
        if not dominated:
            non_dominated.append(row["arm"])
    report = {
        "schema_version": "retention-replay-pareto-report-v1",
        "scope": "development_pareto_not_certification",
        "plan": plan,
        "references": {
            "base_target_nll": base_target,
            "stageA_target_nll": stagea_target,
            "base_external_nll": base_external,
            "stageA_external_nll": stagea_external,
        },
        "rows": rows,
        "non_dominated_arms": non_dominated,
        "joint_pass_arms": [
            row["arm"]
            for row in rows
            if row["target_improvement_vs_stageA"] > 0.0 and row["external_regression_vs_base"] <= 0.0
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(experiment_dir / f"{output_stem}.json", report)
    lines = [
        "# Retention Replay Pareto Development Report",
        "",
        "| Arm | Target fraction | Target NLL | Target gain vs Stage-A | External NLL | External regression vs base |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['arm']}` | {row['target_fraction']:.2f} | {row['target_mean_nll']:.9f} | "
            f"{row['target_improvement_vs_stageA']:.9f} | {row['external_mean_nll']:.9f} | "
            f"{row['external_regression_vs_base']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"Non-dominated development arms: {', '.join(f'`{arm}`' for arm in non_dominated)}",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (experiment_dir / f"{output_stem}.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retention replay Pareto report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs") / "retention_replay_development_plan_qwen25_0p5b_fineweb.json",
    )
    parser.add_argument("--output-stem", default="retention_replay_pareto_report")
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan, str(args.output_stem))
    print({"non_dominated_arms": report["non_dominated_arms"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
