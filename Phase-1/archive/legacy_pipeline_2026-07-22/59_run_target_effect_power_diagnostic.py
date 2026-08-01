#!/usr/bin/env python3
"""Run paired block-level target-effect power diagnostics on fresh seed models."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path("configs") / "retention_recipe_confirmatory_plan_qwen25_0p5b_fineweb.json"


@torch.no_grad()
def _block_losses(model_path: Path, blocks_path: Path) -> List[float]:
    from transformers import AutoModelForCausalLM

    runner = __import__("37_run_slm_update_training")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(runner.BlockDataset(blocks_path), batch_size=1, shuffle=False)
    losses: List[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        output = model(input_ids=input_ids, labels=input_ids)
        losses.append(float(output.loss.detach().cpu()))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return losses


def _paired_summary(stagea: List[float], candidate: List[float], rounds: int, seed: int) -> Dict[str, Any]:
    if len(stagea) != len(candidate) or not stagea:
        raise ValueError("Paired block losses must be nonempty and equal length.")
    deltas = np.asarray(stagea, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(rounds, dtype=np.float64)
    for idx in range(rounds):
        sample = rng.integers(0, len(deltas), size=len(deltas))
        bootstrap[idx] = float(np.mean(deltas[sample]))
    mean = float(np.mean(deltas))
    stdev = float(np.std(deltas, ddof=1))
    standard_error = stdev / math.sqrt(len(deltas))
    return {
        "paired_blocks": len(deltas),
        "mean_target_improvement": mean,
        "paired_block_stdev": stdev,
        "standard_error_normal_approx": standard_error,
        "minimum_detectable_effect_95_normal_approx": 1.96 * standard_error,
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "candidate_block_win_rate": float(np.mean(deltas > 0.0)),
        "candidate_block_tie_rate": float(np.mean(deltas == 0.0)),
        "delta_quantiles": {
            "p01": float(np.quantile(deltas, 0.01)),
            "p10": float(np.quantile(deltas, 0.10)),
            "p25": float(np.quantile(deltas, 0.25)),
            "p50": float(np.quantile(deltas, 0.50)),
            "p75": float(np.quantile(deltas, 0.75)),
            "p90": float(np.quantile(deltas, 0.90)),
            "p99": float(np.quantile(deltas, 0.99)),
        },
        "effect_over_mde95": mean / (1.96 * standard_error) if standard_error > 0.0 else None,
    }


def run_diagnostic(experiment_dir: Path, plan_path: Path, rounds: int) -> Dict[str, Any]:
    plan = load_json(plan_path)
    blocks_path = experiment_dir / "token_blocks_retention_confirmatory" / "retention_confirmatory_target.pt"
    seed_rows = []
    for seed in plan["fresh_training_seeds"]:
        stagea_path = experiment_dir / "model_runs" / f"retention_confirm_seed{seed}_stageA_random_equal_budget"
        candidate_path = experiment_dir / "model_runs" / f"retention_confirm_seed{seed}_retention_replay_target099"
        stagea_losses = _block_losses(stagea_path, blocks_path)
        candidate_losses = _block_losses(candidate_path, blocks_path)
        summary = _paired_summary(stagea_losses, candidate_losses, rounds, int(seed))
        summary["seed"] = int(seed)
        seed_rows.append(summary)

    seed_effects = [float(row["mean_target_improvement"]) for row in seed_rows]
    report = {
        "schema_version": "target-effect-power-diagnostic-v1",
        "scope": "paired_block_noise_floor_diagnostic_not_certification",
        "plan": str(plan_path),
        "eval_blocks": str(blocks_path),
        "bootstrap_rounds": rounds,
        "seed_rows": seed_rows,
        "cross_seed_summary": {
            "seed_count": len(seed_effects),
            "mean_target_improvement": float(np.mean(seed_effects)),
            "sample_stdev": float(np.std(seed_effects, ddof=1)) if len(seed_effects) > 1 else 0.0,
            "positive_seed_count": sum(1 for value in seed_effects if value > 0.0),
            "minimum_seed_effect": min(seed_effects),
            "maximum_seed_effect": max(seed_effects),
        },
        "interpretation": (
            "Paired block-level intervals diagnose evaluation noise conditional on each trained model pair. "
            "Cross-training-seed instability remains the decisive limitation."
        ),
        "limitations": [
            "Packed adjacent token blocks are not guaranteed independent.",
            "Only two fresh training seeds are available.",
            "Block-level confidence intervals do not replace training-seed replication.",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(experiment_dir / "target_effect_power_diagnostic.json", report)
    lines = [
        "# Target Effect Power Diagnostic",
        "",
        "| Seed | Mean gain | Paired MDE95 | CI95 low | CI95 high | Block win rate | Effect / MDE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['mean_target_improvement']:.9f} | "
            f"{row['minimum_detectable_effect_95_normal_approx']:.9f} | "
            f"{row['bootstrap_ci95_low']:.9f} | {row['bootstrap_ci95_high']:.9f} | "
            f"{row['candidate_block_win_rate']:.6f} | {row['effect_over_mde95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    (experiment_dir / "target_effect_power_diagnostic.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired target-effect power diagnostic.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    args = parser.parse_args()
    report = run_diagnostic(args.experiment_dir, args.plan, int(args.bootstrap_rounds))
    print({"cross_seed_summary": report["cross_seed_summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
