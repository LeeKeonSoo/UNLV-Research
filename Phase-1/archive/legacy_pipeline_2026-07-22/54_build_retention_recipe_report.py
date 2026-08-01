#!/usr/bin/env python3
"""Build matched training-recipe target versus retention development evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path("configs") / "retention_recipe_development_plan_qwen25_0p5b_fineweb.json"


def _nll(path: Path) -> float:
    value = load_json(path).get("mean_nll")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing mean_nll: {path}")
    return float(value)


def build_report(experiment_dir: Path, plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_dir = experiment_dir / "eval_results"
    target_eval = "confirmatory_coverage_stratified_stageA_eval"
    base_external = _nll(eval_dir / "guardrail_base_wikitext103.json")
    rows = []
    joint_pass = []
    for recipe in plan["recipes"]:
        recipe_id = str(recipe["recipe_id"])
        stagea_target = _nll(eval_dir / f"recipe_{recipe_id}_stageA_{target_eval}.json")
        stagea_external = _nll(eval_dir / f"recipe_{recipe_id}_stageA_wikitext103.json")
        for arm in ("retention_replay_target100", "retention_replay_target099"):
            target_nll = _nll(eval_dir / f"recipe_{recipe_id}_{arm}_{target_eval}.json")
            external_nll = _nll(eval_dir / f"recipe_{recipe_id}_{arm}_wikitext103.json")
            target_gain = stagea_target - target_nll
            external_regression = external_nll - base_external
            passed = target_gain > 0.0 and external_regression <= 0.0
            row = {
                "recipe_id": recipe_id,
                "learning_rate": float(recipe["learning_rate"]),
                "optimizer_steps": int(recipe["optimizer_steps"]),
                "arm": arm,
                "stageA_target_nll": stagea_target,
                "stageA_external_nll": stagea_external,
                "target_nll": target_nll,
                "target_improvement_vs_matched_stageA": target_gain,
                "external_nll": external_nll,
                "external_regression_vs_base": external_regression,
                "joint_pass": passed,
            }
            rows.append(row)
            if passed:
                joint_pass.append({"recipe_id": recipe_id, "arm": arm})
    report = {
        "schema_version": "retention-recipe-development-report-v1",
        "scope": "development_training_recipe_not_certification",
        "plan": plan,
        "base_external_nll": base_external,
        "rows": rows,
        "joint_pass_candidates": joint_pass,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(experiment_dir / "retention_recipe_development_report.json", report)
    lines = [
        "# Retention Training Recipe Development Report",
        "",
        "| Recipe | Arm | Target gain vs matched Stage-A | External regression vs base | Joint pass |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['recipe_id']}` | `{row['arm']}` | "
            f"{row['target_improvement_vs_matched_stageA']:.9f} | "
            f"{row['external_regression_vs_base']:.9f} | {row['joint_pass']} |"
        )
    lines.extend(
        [
            "",
            f"Joint-pass candidates: {joint_pass or 'none'}",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    (experiment_dir / "retention_recipe_development_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retention recipe development report.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan)
    print({"joint_pass_candidates": report["joint_pass_candidates"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
