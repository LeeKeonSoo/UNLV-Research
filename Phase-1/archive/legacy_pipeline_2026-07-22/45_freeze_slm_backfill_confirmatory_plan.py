#!/usr/bin/env python3
"""Freeze the post-exploratory coverage-backfill confirmatory protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "configs" / "slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _file_sha256(path), "bytes": path.stat().st_size}


def freeze_plan(experiment_dir: Path, output_path: Path) -> Dict[str, Any]:
    frozen_training = load_json(experiment_dir / "frozen_training_plan.json")
    backfill = load_json(experiment_dir / "coverage_backfilled_interleaved50_equal_budget_manifest.json")
    holdouts = load_json(experiment_dir / "confirmatory_holdouts_manifest.json")
    if not bool((holdouts.get("disjointness") or {}).get("exact_uid_disjoint")):
        raise RuntimeError("Confirmatory holdouts are not exactly disjoint.")
    arm_name = str(backfill.get("arm_name") or "")
    arm_path = Path(str(backfill.get("path") or ""))
    stagea_path = Path(str(((frozen_training.get("arm_token_counts") or {}).get("stageA_random_equal_budget") or {}).get("path") or ""))
    selected_path = Path(str(((frozen_training.get("arm_token_counts") or {}).get("curated_equal_budget") or {}).get("path") or ""))
    broad_path = Path(str(((holdouts.get("holdouts") or {}).get("confirmatory_broad_stageA_eval") or {}).get("path") or ""))
    stratified_path = Path(
        str(((holdouts.get("holdouts") or {}).get("confirmatory_coverage_stratified_stageA_eval") or {}).get("path") or "")
    )
    plan = {
        "schema_version": "slm-backfill-confirmatory-plan-v1",
        "plan_name": "confirm_qwen25_0p5b_fineweb_coverage_backfill_interleaved50_v1",
        "status": "frozen_before_confirmatory_training_outcomes",
        "frozen_date": "2026-06-10",
        "experiment_dir": str(experiment_dir),
        "research_question": "Does the frozen 50/50 selected-core plus Stage-A coverage-backfill arm improve broad untouched heldout NLL over equal-budget Stage-A random?",
        "framework_scope": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selected core; no Utility objective",
            "release_training_construction": "frozen selected-core plus Stage-A coverage support mixture",
            "stage_c": "subset-level target-SLM outcome validation",
            "utility_scope": "Stage C only; never selector objective",
        },
        "target_model": frozen_training.get("target_model"),
        "frozen_candidate": {
            "arm": arm_name,
            "selected_core_fraction": backfill.get("selected_core_fraction"),
            "ordering_policy": backfill.get("ordering_policy"),
            "target_tokens": backfill.get("target_tokens"),
            "file": _frozen_file(arm_path),
            "manifest": _frozen_file(experiment_dir / "coverage_backfilled_interleaved50_equal_budget_manifest.json"),
        },
        "comparators": {
            "primary": {
                "arm": "stageA_random_equal_budget",
                "file": _frozen_file(stagea_path),
            },
            "mechanism_diagnostic": {
                "arm": "curated_equal_budget",
                "file": _frozen_file(selected_path),
                "required_new_training_runs": False,
            },
            "base_no_update": {"required_training_run": False},
        },
        "confirmatory_seeds": [20260609, 20260610],
        "excluded_from_confirmatory_success_count": {
            "seed": 20260608,
            "reason": "The 50/50 candidate was created after observing the seed-20260608 selected-only full-budget reversal.",
        },
        "training": {
            "sequence_length": int((frozen_training.get("token_budget") or {}).get("sequence_length") or 1024),
            "matched_target_tokens": int(backfill.get("target_tokens") or 0),
            "learning_rate": 0.00001,
            "weight_decay": 0.1,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "dtype": "fp16",
            "train_mode": "full",
            "max_steps": None,
            "required_new_runs": [
                f"{arm_name}_seed20260609",
                "stageA_random_equal_budget_seed20260609",
                f"{arm_name}_seed20260610",
                "stageA_random_equal_budget_seed20260610",
            ],
        },
        "evaluation": {
            "primary": {
                "name": "confirmatory_broad_stageA_eval",
                "metric": "mean_nll",
                "direction": "lower_is_better",
                "file": _frozen_file(broad_path),
                "role": "Only this evaluation determines primary success.",
            },
            "secondary": {
                "name": "confirmatory_coverage_stratified_stageA_eval",
                "metric": "mean_nll",
                "direction": "lower_is_better",
                "file": _frozen_file(stratified_path),
                "role": "Mechanism diagnostic; cannot rescue primary failure.",
            },
            "holdout_manifest": _frozen_file(experiment_dir / "confirmatory_holdouts_manifest.json"),
        },
        "primary_success_rule": {
            "required": [
                "backfilled mean NLL is lower than Stage-A random mean NLL on the primary holdout across the two fresh confirmatory seeds",
                "backfilled NLL is lower than Stage-A random NLL on both fresh confirmatory seeds",
                "no missing or NaN primary outcomes",
                "training recipe and frozen files match this plan",
            ],
            "failure_rule": "Any required condition failure means the frozen 50/50 confirmatory direction is not supported.",
            "secondary_evaluation_rule": "Secondary outcomes are reported but cannot change primary pass/fail.",
            "statistical_boundary": "Two fresh seeds provide directional replication, not a final high-power confidence-interval claim.",
        },
        "claim_boundary": {
            "allowed_if_successful": "Scoped confirmatory evidence that the frozen 50/50 coverage-backfilled release arm improves internal broad heldout NLL over equal-budget Stage-A random for this FineWeb-Edu/Qwen setup.",
            "not_allowed": [
                "Claiming that 50/50 is optimal or universal",
                "Universal raw-corpus curation claim",
                "Dataset-independent or target-model-independent Utility improvement",
                "Deployment-ready claim without external benchmark, forgetting, safety, and contamination checks",
                "Using target-SLM outcomes as a Stage-B selector objective",
            ],
        },
    }
    save_json(output_path, plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze coverage-backfill confirmatory protocol.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = freeze_plan(args.experiment_dir, args.output)
    print(json.dumps({"status": plan["status"], "plan": str(args.output), "seeds": plan["confirmatory_seeds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
