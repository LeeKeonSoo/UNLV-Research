#!/usr/bin/env python3
"""Freeze the first joint-pass retention-aware recipe as a confirmatory candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_OUTPUT = Path("configs") / "retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json"


def freeze_plan(experiment_dir: Path) -> Dict[str, Any]:
    report_path = experiment_dir / "retention_recipe_development_report.json"
    report = load_json(report_path)
    required_candidate = {"recipe_id": "lr5e6_s128", "arm": "retention_replay_target099"}
    if required_candidate not in report.get("joint_pass_candidates", []):
        raise RuntimeError("Required development joint-pass candidate is not supported.")
    candidate_path = experiment_dir / "retention_replay_target099.jsonl"
    comparator_path = experiment_dir / "stageA_random_equal_budget.jsonl"
    source_plans = [
        Path("configs") / "retention_recipe_development_plan_qwen25_0p5b_fineweb.json",
        Path("configs") / "retention_replay_boundary_plan_qwen25_0p5b_fineweb.json",
    ]
    plan = {
        "schema_version": "retention-recipe-confirmatory-plan-v1",
        "plan_name": "fineweb_retention_recipe_joint_confirmatory_v1",
        "candidate": {
            "arm": "retention_replay_target099",
            "target_fraction": 0.99,
            "general_replay_fraction": 0.01,
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "training_recipe": {
                "learning_rate": 0.000005,
                "optimizer_steps": 128,
                "sequence_length": 1024,
                "train_sequences": 1024,
                "gradient_accumulation_steps": 4,
                "full_parameter_training": True,
            },
        },
        "matched_comparator": {
            "arm": "stageA_random_equal_budget",
            "path": str(comparator_path),
            "sha256": sha256_file(comparator_path),
            "uses_identical_training_recipe": True,
        },
        "fresh_training_seeds": [20260612, 20260613],
        "required_new_evidence_before_training": [
            "untouched target-distribution holdout not used in replay-ratio or recipe development",
            "untouched external-retention holdout not used in replay-ratio or recipe development",
            "exact and near-duplicate train/evaluation overlap audit",
        ],
        "per_seed_success_rule": (
            "candidate target NLL must be lower than recipe-matched Stage-A random, "
            "and candidate external-retention NLL must not exceed base no-update"
        ),
        "overall_success_rule": "Both fresh seeds must satisfy the per-seed joint rule.",
        "stop_rule": "Stop after the first fresh seed that fails either required outcome.",
        "forbidden_changes_after_freeze": [
            "candidate replay ratio",
            "learning rate",
            "optimizer steps",
            "training sequence budget",
            "matched comparator",
            "success rule",
            "evaluation holdouts after results are observed",
        ],
        "development_evidence": {
            "report": str(report_path),
            "sha256": sha256_file(report_path),
            "candidate_target_improvement_vs_matched_stageA": 0.0001529223235077204,
            "candidate_external_regression_vs_base": -0.00016911544146092083,
            "role": "candidate-selection evidence only; excluded from confirmatory success count",
        },
        "source_plan_hashes": {str(path): sha256_file(path) for path in source_plans},
        "framework_scope": {
            "stage_b": "unchanged; no Utility or target-model objective",
            "stage_c": "joint target and retention validation",
            "release_layer": "candidate construction and deployment-contract decision",
            "utility_scope": "Stage C validation only; never selector objective",
        },
        "claim_boundary": (
            "Frozen confirmatory candidate only. No certification or deployment claim until "
            "fresh seeds and untouched evaluations satisfy the frozen joint rule."
        ),
    }
    holdout_manifest_path = experiment_dir / "retention_confirmatory_holdouts_manifest.json"
    if holdout_manifest_path.exists():
        holdout_manifest = load_json(holdout_manifest_path)
        plan["confirmatory_holdouts"] = {
            "manifest": {"path": str(holdout_manifest_path), "sha256": sha256_file(holdout_manifest_path)},
            "target": holdout_manifest["target_holdout"],
            "external": holdout_manifest["external_holdout"],
            "status": "bound_before_confirmatory_training",
        }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze retention recipe confirmatory candidate.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = freeze_plan(args.experiment_dir)
    save_json(args.output, plan)
    print({"candidate": plan["candidate"]["arm"], "fresh_seeds": plan["fresh_training_seeds"], "output": str(args.output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
