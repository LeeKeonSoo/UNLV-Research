#!/usr/bin/env python3
"""Cross-evaluate matched trained model pairs on development and fresh target holdouts."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_eval_common import OUTPUT_DIR, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"


MODEL_PAIRS = {
    20260611: {
        "stageA": "recipe_lr5e6_s128_stageA_random_equal_budget_seed20260611",
        "candidate": "recipe_lr5e6_s128_retention_replay_target099_seed20260611",
        "role": "development_recipe_seed",
    },
    20260612: {
        "stageA": "retention_confirm_seed20260612_stageA_random_equal_budget",
        "candidate": "retention_confirm_seed20260612_retention_replay_target099",
        "role": "fresh_confirmatory_seed",
    },
    20260613: {
        "stageA": "retention_confirm_seed20260613_stageA_random_equal_budget",
        "candidate": "retention_confirm_seed20260613_retention_replay_target099",
        "role": "fresh_confirmatory_seed",
    },
}


@torch.no_grad()
def _losses(model_path: Path, blocks_path: Path) -> List[float]:
    from transformers import AutoModelForCausalLM

    runner = __import__("37_run_slm_update_training")
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    values = []
    for batch in DataLoader(runner.BlockDataset(blocks_path), batch_size=1, shuffle=False):
        input_ids = batch["input_ids"].to(device)
        values.append(float(model(input_ids=input_ids, labels=input_ids).loss.detach().cpu()))
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return values


def _paired(stagea: List[float], candidate: List[float], rounds: int, seed: int) -> Dict[str, Any]:
    deltas = np.asarray(stagea, dtype=np.float64) - np.asarray(candidate, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(rounds, dtype=np.float64)
    for idx in range(rounds):
        boot[idx] = float(np.mean(deltas[rng.integers(0, len(deltas), size=len(deltas))]))
    return {
        "paired_blocks": len(deltas),
        "target_improvement_vs_matched_stageA": float(np.mean(deltas)),
        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
        "candidate_block_win_rate": float(np.mean(deltas > 0.0)),
    }


def run(experiment_dir: Path, output: Path, rounds: int) -> Dict[str, Any]:
    holdouts = {
        "development_target": experiment_dir / "token_blocks_full" / "confirmatory_coverage_stratified_stageA_eval.pt",
        "fresh_confirmatory_target": experiment_dir
        / "token_blocks_retention_confirmatory"
        / "retention_confirmatory_target.pt",
    }
    train_dir_a = experiment_dir / "token_blocks_retention_replay_pilot"
    train_dir_b = experiment_dir / "token_blocks_retention_confirmatory"
    train_blocks_identical = {
        arm: bool(
            torch.equal(
                torch.load(train_dir_a / f"{arm}.pt", map_location="cpu")["input_ids"],
                torch.load(train_dir_b / f"{arm}.pt", map_location="cpu")["input_ids"],
            )
        )
        for arm in ("stageA_random_equal_budget", "retention_replay_target099")
    }

    rows = []
    for seed, pair in MODEL_PAIRS.items():
        for holdout_name, blocks_path in holdouts.items():
            stagea = _losses(experiment_dir / "model_runs" / pair["stageA"], blocks_path)
            candidate = _losses(experiment_dir / "model_runs" / pair["candidate"], blocks_path)
            row = _paired(stagea, candidate, rounds, seed)
            row.update({"seed": seed, "seed_role": pair["role"], "holdout": holdout_name})
            rows.append(row)
            print(row)

    by_seed = []
    for seed in MODEL_PAIRS:
        values = {row["holdout"]: row for row in rows if row["seed"] == seed}
        development = float(values["development_target"]["target_improvement_vs_matched_stageA"])
        confirmatory = float(values["fresh_confirmatory_target"]["target_improvement_vs_matched_stageA"])
        by_seed.append(
            {
                "seed": seed,
                "development_target_improvement": development,
                "fresh_confirmatory_target_improvement": confirmatory,
                "holdout_shift_delta": confirmatory - development,
                "sign_consistent": (development > 0.0) == (confirmatory > 0.0),
            }
        )
    report = {
        "schema_version": "target-holdout-shift-diagnostic-v1",
        "scope": "development_cross_holdout_diagnostic_not_certification",
        "train_blocks_identical": train_blocks_identical,
        "rows": rows,
        "by_seed": by_seed,
        "interpretation_rule": (
            "If a fixed trained model pair changes sign across target holdouts, target-distribution "
            "generalization is a separate limitation from training-seed instability."
        ),
        "limitations": [
            "Only three trained model pairs are available.",
            "Packed adjacent block bootstrap intervals do not replace training-seed replication.",
            "The development target holdout was previously used in recipe development.",
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Diagnostic evidence only; no deployment or certification claim.",
    }
    save_json(output, report)
    lines = [
        "# Target Holdout Shift Diagnostic",
        "",
        "| Seed | Role | Development gain | Fresh confirmatory gain | Shift delta | Sign consistent |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in by_seed:
        lines.append(
            f"| {row['seed']} | {MODEL_PAIRS[row['seed']]['role']} | "
            f"{row['development_target_improvement']:.9f} | "
            f"{row['fresh_confirmatory_target_improvement']:.9f} | "
            f"{row['holdout_shift_delta']:.9f} | {row['sign_consistent']} |"
        )
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-target-holdout diagnostics.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "target_holdout_shift_diagnostic.json",
    )
    parser.add_argument("--bootstrap-rounds", type=int, default=3000)
    args = parser.parse_args()
    report = run(args.experiment_dir, args.output, int(args.bootstrap_rounds))
    print({"by_seed": report["by_seed"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
