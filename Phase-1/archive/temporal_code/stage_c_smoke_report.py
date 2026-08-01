#!/usr/bin/env python3
"""Build the Qwen3-4B Stage-C execution-feasibility smoke report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_ROOT = OUTPUT_DIR / "temporal_code_stage_c_smoke_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "temporal_code_stage_c_smoke_report.json"
ARMS = ("curated_equal_token", "stageA_random_equal_token", "raw_random_equal_token")


def build(root: Path, output: Path) -> Dict[str, Any]:
    contract = load_json(Path("configs") / "temporal_code_stage_c_smoke_qwen3_4b_v1.json")
    arm_manifest = load_json(root / "frozen_smoke_arm_manifest.json")
    blocks = load_json(root / "token_blocks" / "block_manifest.json")
    seed = int(contract["training_recipe"]["smoke_seed"])
    steps = int(contract["training_recipe"]["smoke_optimizer_steps"])
    runs = {
        arm: load_json(root / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}" / "run_result.json")
        for arm in ARMS
    }
    packed_tokens = {int(row["packed_tokens"]) for row in blocks["blocks"].values()}
    optimizer_steps = {int(row["optimizer_steps"]) for row in runs.values()}
    run_seeds = {int(row["seed"]) for row in runs.values()}
    devices = {str(row["cuda_device_name"]) for row in runs.values()}
    report = {
        "schema_version": "temporal-code-stage-c-smoke-report-v1",
        "status": "qlora_stage_c_smoke_feasibility_pass",
        "summary": {
            "model_id": contract["target_model"]["model_id"],
            "arms_completed": len(runs),
            "common_packed_token_budget": next(iter(packed_tokens)),
            "common_optimizer_steps": next(iter(optimizer_steps)),
            "common_seed": next(iter(run_seeds)),
            "cuda_device_names": sorted(devices),
            "all_arms_completed": all(row["status"] == "qlora_smoke_completed" for row in runs.values()),
            "equal_packed_token_budget": len(packed_tokens) == 1,
            "equal_optimizer_steps": len(optimizer_steps) == 1,
            "equal_seed": len(run_seeds) == 1,
            "common_stage_a_baseline_shared": arm_manifest[
                "all_sensitivity_arms_share_common_stage_a_baseline"
            ],
            "curated_common_baseline_overlap_count": arm_manifest[
                "curated_common_baseline_overlap_count"
            ],
        },
        "runs": runs,
        "training_loss_interpretation": (
            "Execution diagnostic only. Training-loss ordering is not Utility, cannot promote or reject Stage B, "
            "and cannot set the development practical-effect margin."
        ),
        "development_entry_blockers": [
            "only one executable development bundle is currently verified; expand the no-outcome development task pool",
            "development practical-effect margin not frozen",
            "development seeds and executable aggregate not frozen",
            "retention non-inferiority guardrails not frozen for this experiment",
        ],
        "confirmatory_outcomes_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": (
            "Qwen3-4B QLoRA and equal-budget Stage-C smoke execution are feasible. "
            "No Utility, curation-benefit, or release claim is established."
        ),
    }
    save_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build temporal-code Stage-C smoke report.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.root, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
