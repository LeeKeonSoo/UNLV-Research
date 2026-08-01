#!/usr/bin/env python3
"""Build a report for code-domain Qwen3-4B QLoRA smoke feasibility."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_CONFIG = Path("configs") / "code_domain_qlora_smoke_qwen3_4b_v1.json"
DEFAULT_SMOKE_DIR = OUTPUT_DIR / "code_domain_qlora_smoke_qwen3_4b_v1"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_qlora_smoke_qwen3_4b_report.json"


def _run_result_path(smoke_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return smoke_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}" / "run_result.json"


def build(config_path: Path, smoke_dir: Path, output_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    recipe = config["training_recipe"]
    seed = int(recipe["seed"])
    steps = int(recipe["smoke_optimizer_steps"])
    block_manifest_path = smoke_dir / "token_blocks" / "block_manifest.json"
    blocks = load_json(block_manifest_path)
    runs: Dict[str, Any] = {}
    blockers = []
    for arm in config["arms"]:
        run_path = _run_result_path(smoke_dir, arm, seed, steps)
        if not run_path.exists():
            blockers.append(f"missing_run_result:{arm}")
            continue
        run = load_json(run_path)
        runs[arm] = run
        if run.get("status") != "qlora_smoke_completed":
            blockers.append(f"run_not_completed:{arm}")
        if int(run.get("optimizer_steps") or 0) != steps:
            blockers.append(f"optimizer_step_mismatch:{arm}")
        if run.get("device_summary", {}).get("device") != "cuda":
            blockers.append(f"run_not_cuda:{arm}")
    packed_budgets = {
        arm: int(row["packed_tokens"])
        for arm, row in (blocks.get("blocks") or {}).items()
    }
    if len(set(packed_budgets.values())) != 1:
        blockers.append("packed_token_budget_mismatch")
    trainable_counts = {
        arm: int(row.get("trainable_parameters") or 0)
        for arm, row in runs.items()
    }
    if len(set(trainable_counts.values())) != 1:
        blockers.append("trainable_parameter_mismatch")
    device_names = {
        arm: (row.get("device_summary", {}).get("gpus") or [{}])[0].get("name")
        for arm, row in runs.items()
    }
    report = {
        "schema_version": "code-domain-qlora-smoke-report-v1",
        "status": "qlora_smoke_feasible" if not blockers else "qlora_smoke_blocked",
        "config_sha256": sha256_file(config_path),
        "block_manifest_sha256": sha256_file(block_manifest_path),
        "summary": {
            "arm_count": len(config["arms"]),
            "completed_run_count": len(runs),
            "optimizer_steps_per_arm": steps,
            "gradient_accumulation_steps": int(recipe["gradient_accumulation_steps"]),
            "common_packed_token_budget": blocks["common_packed_token_budget"],
            "training_token_budget_cap": blocks["training_token_budget_cap"],
            "trainable_parameters": next(iter(trainable_counts.values())) if trainable_counts else None,
            "device_names": device_names,
            "blockers": blockers,
        },
        "arms": {
            arm: {
                "packed_tokens": packed_budgets.get(arm),
                "blocks": (blocks.get("blocks") or {}).get(arm, {}).get("blocks"),
                "optimizer_steps": runs.get(arm, {}).get("optimizer_steps"),
                "micro_steps": runs.get(arm, {}).get("micro_steps"),
                "mean_microbatch_loss": runs.get(arm, {}).get("mean_microbatch_loss"),
                "elapsed_seconds": runs.get(arm, {}).get("elapsed_seconds"),
                "cuda_peak_memory_allocated": runs.get(arm, {}).get("cuda_peak_memory_allocated"),
                "run_result_sha256": sha256_file(_run_result_path(smoke_dir, arm, seed, steps))
                if _run_result_path(smoke_dir, arm, seed, steps).exists()
                else None,
            }
            for arm in config["arms"]
        },
        "success_criteria": config["success_criteria"],
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain QLoRA smoke report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(args.config, args.smoke_dir, args.output)
    print({"status": report["status"], **report["summary"]})
    return 0 if report["status"] == "qlora_smoke_feasible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
