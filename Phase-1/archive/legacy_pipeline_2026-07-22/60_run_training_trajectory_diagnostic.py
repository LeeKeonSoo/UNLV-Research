#!/usr/bin/env python3
"""Measure matched Stage-A and replay-candidate development curves."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path("configs") / "retention_training_trajectory_plan_qwen25_0p5b_fineweb.json"


@torch.no_grad()
def _evaluate(model: Any, blocks_path: Path, device: torch.device, dtype: str) -> Dict[str, Any]:
    runner = __import__("37_run_slm_update_training")
    loader = DataLoader(runner.BlockDataset(blocks_path), batch_size=1, shuffle=False)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        autocast_dtype = torch.bfloat16 if dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=(device.type == "cuda" and dtype in {"bf16", "fp16"}),
        ):
            loss = model(input_ids=input_ids, labels=input_ids).loss
        tokens = int(input_ids.numel())
        total_loss += float(loss.detach().cpu()) * tokens
        total_tokens += tokens
    if was_training:
        model.train()
    mean_nll = total_loss / max(1, total_tokens)
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else None,
        "tokens": total_tokens,
        "blocks": len(loader.dataset),
    }


def _train_curve(plan: Dict[str, Any], arm: str, seed: int) -> Dict[str, Any]:
    runner = __import__("37_run_slm_update_training")
    runner._set_seed(seed)
    recipe = plan["training_recipe"]
    blocks = plan["blocks"]
    train_path = Path(blocks["train_dir"]) / f"{arm}.pt"
    target_path = Path(blocks["target_eval"])
    external_path = Path(blocks["external_eval"])
    for path in (train_path, target_path, external_path):
        if not path.exists():
            raise FileNotFoundError(path)

    frozen_plan = load_json(DEFAULT_EXPERIMENT_DIR / "frozen_training_plan.json")
    model = runner._load_model(frozen_plan, local_files_only=True, dtype=str(recipe["dtype"]))
    runner._configure_trainable_params(model, "full", 2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = runner.BlockDataset(train_path)
    loader = DataLoader(dataset, batch_size=int(recipe["batch_size"]), shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    grad_accum = int(recipe["gradient_accumulation_steps"])
    max_steps = int(recipe["optimizer_steps"])
    checkpoints = {int(value) for value in plan["checkpoint_steps"]}
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and recipe["dtype"] == "fp16"))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    rows: List[Dict[str, Any]] = []
    step = 0
    micro_step = 0
    started = time.time()
    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            dtype = str(recipe["dtype"])
            autocast_dtype = torch.bfloat16 if dtype == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=(device.type == "cuda" and dtype in {"bf16", "fp16"}),
            ):
                loss = model(input_ids=input_ids, labels=input_ids).loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(float(loss.detach().cpu()) * grad_accum)
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(recipe["max_grad_norm"]),
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step in checkpoints:
                row = {
                    "step": step,
                    "target": _evaluate(model, target_path, device, dtype),
                    "external": _evaluate(model, external_path, device, dtype),
                }
                rows.append(row)
                print(json.dumps({"arm": arm, "seed": seed, **row}))
            if step >= max_steps:
                break

    result = {
        "arm": arm,
        "seed": seed,
        "train_blocks": str(train_path),
        "curve": rows,
        "mean_training_loss": sum(losses) / len(losses),
        "elapsed_seconds": round(time.time() - started, 3),
        "device_summary": runner._visible_device_summary(),
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _build_summary(plan: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_external = float(load_json(Path(plan["base_external_eval_result"]))["mean_nll"])
    by_key = {(int(run["seed"]), str(run["arm"])): run for run in runs}
    rows = []
    for seed in plan["development_training_seeds"]:
        stagea = by_key.get((int(seed), "stageA_random_equal_budget"))
        candidate = by_key.get((int(seed), "retention_replay_target099"))
        if not stagea or not candidate:
            continue
        stagea_steps = {int(row["step"]): row for row in stagea["curve"]}
        candidate_steps = {int(row["step"]): row for row in candidate["curve"]}
        for step in plan["checkpoint_steps"]:
            stagea_row = stagea_steps[int(step)]
            candidate_row = candidate_steps[int(step)]
            target_gain = float(stagea_row["target"]["mean_nll"]) - float(candidate_row["target"]["mean_nll"])
            external_regression = float(candidate_row["external"]["mean_nll"]) - base_external
            rows.append(
                {
                    "seed": int(seed),
                    "step": int(step),
                    "target_improvement_vs_matched_stageA": target_gain,
                    "candidate_external_regression_vs_base": external_regression,
                    "target_pass": target_gain > 0.0,
                    "external_pass": external_regression <= 0.0,
                    "joint_pass": target_gain > 0.0 and external_regression <= 0.0,
                }
            )
    step_summary = []
    for step in plan["checkpoint_steps"]:
        values = [row for row in rows if row["step"] == int(step)]
        gains = [float(row["target_improvement_vs_matched_stageA"]) for row in values]
        step_summary.append(
            {
                "step": int(step),
                "seed_count": len(values),
                "mean_target_improvement": sum(gains) / len(gains) if gains else None,
                "target_positive_seed_count": sum(1 for row in values if row["target_pass"]),
                "external_pass_seed_count": sum(1 for row in values if row["external_pass"]),
                "joint_pass_seed_count": sum(1 for row in values if row["joint_pass"]),
            }
        )
    return {"base_external_nll": base_external, "paired_rows": rows, "step_summary": step_summary}


def run(plan_path: Path, output_path: Path, seeds: List[int] | None) -> Dict[str, Any]:
    plan = load_json(plan_path)
    selected_seeds = seeds or [int(value) for value in plan["development_training_seeds"]]
    runs: List[Dict[str, Any]] = []
    if output_path.exists():
        existing = load_json(output_path)
        runs = list(existing.get("runs") or [])
    completed = {(int(item["seed"]), str(item["arm"])) for item in runs}
    for seed in selected_seeds:
        for arm in plan["arms"]:
            key = (int(seed), str(arm))
            if key in completed:
                continue
            runs.append(_train_curve(plan, str(arm), int(seed)))
            report = {
                "schema_version": "retention-training-trajectory-diagnostic-v1",
                "scope": plan["scope"],
                "plan": plan,
                "runs": runs,
                "summary": _build_summary(plan, runs),
                "utility_scope": plan["utility_scope"],
                "claim_boundary": plan["claim_boundary"],
            }
            save_json(output_path, report)
    report = {
        "schema_version": "retention-training-trajectory-diagnostic-v1",
        "scope": plan["scope"],
        "plan": plan,
        "runs": runs,
        "summary": _build_summary(plan, runs),
        "utility_scope": plan["utility_scope"],
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(output_path, report)
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Retention Training Trajectory Diagnostic",
        "",
        "| Step | Seeds | Mean target gain | Target-positive | External pass | Joint pass |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]["step_summary"]:
        mean = row["mean_target_improvement"]
        mean_text = f"{mean:.9f}" if isinstance(mean, float) else "incomplete"
        lines.append(
            f"| {row['step']} | {row['seed_count']} | {mean_text} | "
            f"{row['target_positive_seed_count']} | {row['external_pass_seed_count']} | "
            f"{row['joint_pass_seed_count']} |"
        )
    lines.extend(["", "## Claim Boundary", "", plan["claim_boundary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run development-only matched training trajectory diagnostics.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "retention_training_trajectory_diagnostic.json",
    )
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    report = run(args.plan, args.output, args.seeds)
    print(json.dumps(report["summary"]["step_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
