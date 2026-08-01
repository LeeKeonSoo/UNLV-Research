#!/usr/bin/env python3
"""Run frozen Qwen2.5-0.5B QLoRA redundancy proxy arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_BLOCK_MANIFEST = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_packed_blocks_manifest.json"
)
DEFAULT_EVAL_INPUTS = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "redundancy_saturation_proxy_qwen25_0p5b_v1"
)
DEFAULT_AUDIT = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_proxy_runner_audit.json"
)


class SafeTensorBlockDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = load_file(path)
        if set(payload) != {"input_ids"}:
            raise RuntimeError(f"Unexpected tensor keys in {path}: {sorted(payload)}")
        self.blocks = payload["input_ids"].to(torch.long)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        value = self.blocks[index]
        return {"input_ids": value, "labels": value.clone()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_summary() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": 0,
            "gpus": [],
        }
    return {
        "device": "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "current_visible_device": torch.cuda.current_device(),
        "gpus": [
            {
                "visible_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _load_contracts(
    plan_path: Path,
    block_manifest_path: Path,
    eval_inputs_path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    plan = load_json(plan_path)
    blocks = load_json(block_manifest_path)
    eval_inputs = load_json(eval_inputs_path)
    if plan["status"] != "frozen_before_proxy_training_outcomes":
        raise RuntimeError("Proxy training plan is not frozen.")
    if blocks["status"] != "redundancy_proxy_exact_blocks_materialized":
        raise RuntimeError("Proxy token blocks are not frozen.")
    if eval_inputs["status"] != "frozen_before_proxy_training_outcomes":
        raise RuntimeError("Proxy evaluation inputs are not frozen.")
    if blocks["frozen_config"]["sha256"] != sha256_file(plan_path):
        raise RuntimeError("Packed blocks do not match the current frozen plan.")
    if eval_inputs["source_contracts"]["proxy_experiment"]["sha256"] != sha256_file(
        plan_path
    ):
        raise RuntimeError("Evaluation inputs do not match the current frozen plan.")
    if eval_inputs["source_contracts"]["packed_blocks_manifest"][
        "sha256"
    ] != sha256_file(block_manifest_path):
        raise RuntimeError("Evaluation inputs do not match the packed-block manifest.")
    return plan, blocks, eval_inputs


def _permutation(block_count: int, seed: int) -> List[int]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(block_count, generator=generator).tolist()


def _order_sha256(order: Iterable[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in order).encode("ascii")
    ).hexdigest()


def audit_runner(
    plan_path: Path,
    block_manifest_path: Path,
    eval_inputs_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    plan, blocks, eval_inputs = _load_contracts(
        plan_path,
        block_manifest_path,
        eval_inputs_path,
    )
    recipe = plan["training_recipe"]
    grad_accum = int(recipe["gradient_accumulation_steps"])
    optimizer_steps = int(recipe["optimizer_steps"])
    required_micro_steps = grad_accum * optimizer_steps
    arm_rows = {
        name: row
        for name, row in blocks["artifacts"].items()
        if row["role"] == "training_arm"
    }
    arm_names = list(plan["arms"])
    blockers = []
    if set(arm_rows) != set(arm_names):
        blockers.append("runner_arm_set_does_not_match_frozen_blocks")
    seed_contract = {}
    for seed in recipe["seeds"]:
        orders = {}
        for arm in arm_names:
            block_count = int(arm_rows[arm]["blocks"])
            order = _permutation(block_count, int(seed))
            orders[arm] = {
                "block_count": block_count,
                "order_sha256": _order_sha256(order),
                "first_16_indices": order[:16],
                "all_indices_consumed_once": (
                    len(order) == block_count
                    and len(set(order)) == block_count
                    and min(order) == 0
                    and max(order) == block_count - 1
                ),
            }
            if block_count != required_micro_steps:
                blockers.append(
                    f"block_count_not_equal_required_micro_steps:{arm}:seed{seed}"
                )
        if len({row["order_sha256"] for row in orders.values()}) != 1:
            blockers.append(f"arm_shuffle_order_mismatch:seed{seed}")
        seed_contract[str(seed)] = orders

    audit = {
        "schema_version": "redundancy-proxy-qlora-runner-audit-v1",
        "status": (
            "redundancy_proxy_qlora_runner_ready"
            if not blockers
            else "redundancy_proxy_qlora_runner_blocked"
        ),
        "source_sha256": {
            str(plan_path): sha256_file(plan_path),
            str(block_manifest_path): sha256_file(block_manifest_path),
            str(eval_inputs_path): sha256_file(eval_inputs_path),
        },
        "training_contract": {
            "arms": arm_names,
            "seeds": recipe["seeds"],
            "micro_batch_size": recipe["micro_batch_size"],
            "gradient_accumulation_steps": grad_accum,
            "optimizer_steps": optimizer_steps,
            "required_micro_steps": required_micro_steps,
            "sequence_length": plan["tokenization_and_packing"]["sequence_length"],
            "tokens_per_arm": plan["tokenization_and_packing"][
                "exact_train_tokens_per_arm"
            ],
            "single_epoch_exact_consumption": True,
            "same_permutation_for_same_seed_across_arms": True,
        },
        "seed_shuffle_contract": seed_contract,
        "completion_contract": {
            "required_files": [
                "adapter_config.json",
                "adapter_model.safetensors",
                "run_result.json",
                "adapter_manifest.json",
            ],
            "partial_run_reusable": False,
            "completed_run_must_match_plan_block_and_eval_input_hashes": True,
        },
        "evaluation_inputs_status": eval_inputs["status"],
        "blockers": blockers,
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Runner semantics audit only. No model outcome, Utility, retention, "
            "promotion, or release claim."
        ),
    }
    save_json(output_path, audit)
    return audit


def _load_qlora_model(plan: Dict[str, Any]) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    recipe = plan["training_recipe"]
    adapter = recipe["adapter"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        plan["target_model"]["snapshot_path"],
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(recipe["gradient_checkpointing"]),
    )
    lora = LoraConfig(
        r=int(adapter["rank"]),
        lora_alpha=int(adapter["alpha"]),
        lora_dropout=float(adapter["dropout"]),
        target_modules=str(adapter["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def _run_dir(output_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def _adapter_manifest(run_dir: Path) -> Dict[str, Any]:
    files = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing adapter artifact: {path}")
        files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "schema_version": "redundancy-proxy-adapter-manifest-v1",
        "files": files,
    }


def _completed_run(
    output_dir: Path,
    arm: str,
    seed: int,
    steps: int,
    plan_sha256: str,
    block_sha256: str,
    eval_inputs_sha256: str,
) -> bool:
    run_dir = _run_dir(output_dir, arm, seed, steps)
    required = [
        run_dir / "adapter_config.json",
        run_dir / "adapter_model.safetensors",
        run_dir / "run_result.json",
        run_dir / "adapter_manifest.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        result = load_json(run_dir / "run_result.json")
        manifest = load_json(run_dir / "adapter_manifest.json")
    except (OSError, json.JSONDecodeError):
        return False
    if result.get("status") != "redundancy_proxy_qlora_completed":
        return False
    if int(result.get("optimizer_steps") or 0) != steps:
        return False
    if result.get("plan_sha256") != plan_sha256:
        return False
    if result.get("train_blocks_sha256") != block_sha256:
        return False
    if result.get("evaluation_inputs_sha256") != eval_inputs_sha256:
        return False
    for row in manifest.get("files", {}).values():
        path = Path(row["path"])
        if not path.exists() or sha256_file(path) != row["sha256"]:
            return False
    return True


def train_one(
    plan_path: Path,
    block_manifest_path: Path,
    eval_inputs_path: Path,
    output_dir: Path,
    arm: str,
    seed: int,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for redundancy proxy QLoRA training.")
    plan, blocks, _ = _load_contracts(
        plan_path,
        block_manifest_path,
        eval_inputs_path,
    )
    if arm not in plan["arms"]:
        raise ValueError(f"Unknown frozen arm: {arm}")
    if int(seed) not in {int(value) for value in plan["training_recipe"]["seeds"]}:
        raise ValueError(f"Seed is not in frozen contract: {seed}")

    torch.cuda.set_device(0)
    torch.cuda.init()
    recipe = plan["training_recipe"]
    steps = int(recipe["optimizer_steps"])
    grad_accum = int(recipe["gradient_accumulation_steps"])
    block_row = blocks["artifacts"][arm]
    blocks_path = Path(block_row["path"])
    if sha256_file(blocks_path) != block_row["file_sha256"]:
        raise RuntimeError(f"Frozen training block hash mismatch: {blocks_path}")

    _set_seed(seed)
    torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    started = time.time()
    dataset = SafeTensorBlockDataset(blocks_path)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(recipe["micro_batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    model = _load_qlora_model(plan)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable = sum(param.numel() for param in trainable_params)
    total = sum(param.numel() for param in model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: List[float] = []
    optimizer_steps = 0
    micro_steps = 0
    for batch in loader:
        batch = {key: value.to(0) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / grad_accum
        loss.backward()
        losses.append(float(loss.detach().cpu()) * grad_accum)
        micro_steps += 1
        if micro_steps % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                float(recipe["max_grad_norm"]),
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "optimizer_step": optimizer_steps,
                        "mean_recent_loss": sum(losses[-grad_accum:]) / grad_accum,
                    }
                )
            )
    if optimizer_steps != steps or micro_steps != steps * grad_accum:
        raise RuntimeError(
            f"Frozen compute mismatch: optimizer={optimizer_steps}/{steps}, "
            f"micro={micro_steps}/{steps * grad_accum}"
        )

    run_dir = _run_dir(output_dir, arm, seed, steps)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir, safe_serialization=True)
    adapter_manifest = _adapter_manifest(run_dir)
    save_json(run_dir / "adapter_manifest.json", adapter_manifest)
    result = {
        "schema_version": "redundancy-proxy-qlora-run-v1",
        "status": "redundancy_proxy_qlora_completed",
        "arm": arm,
        "seed": int(seed),
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "sequence_length": int(
            plan["tokenization_and_packing"]["sequence_length"]
        ),
        "tokens_consumed": int(
            micro_steps
            * recipe["micro_batch_size"]
            * plan["tokenization_and_packing"]["sequence_length"]
        ),
        "mean_microbatch_loss": sum(losses) / len(losses),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "shuffle_order_sha256": _order_sha256(_permutation(len(dataset), seed)),
        "plan_sha256": sha256_file(plan_path),
        "block_manifest_sha256": sha256_file(block_manifest_path),
        "evaluation_inputs_sha256": sha256_file(eval_inputs_path),
        "train_blocks": str(blocks_path),
        "train_blocks_sha256": block_row["file_sha256"],
        "adapter_manifest_sha256": sha256_file(run_dir / "adapter_manifest.json"),
        "device_summary": _device_summary(),
        "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(run_dir),
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Frozen equal-compute QLoRA training artifact only. Training loss is "
            "diagnostic and cannot establish Utility or promotion."
        ),
    }
    save_json(run_dir / "run_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def train_missing(
    plan_path: Path,
    block_manifest_path: Path,
    eval_inputs_path: Path,
    output_dir: Path,
    arms: List[str],
    seeds: List[int],
    max_runs: int | None,
) -> Dict[str, Any]:
    plan, blocks, _ = _load_contracts(
        plan_path,
        block_manifest_path,
        eval_inputs_path,
    )
    steps = int(plan["training_recipe"]["optimizer_steps"])
    plan_hash = sha256_file(plan_path)
    eval_hash = sha256_file(eval_inputs_path)
    executed = []
    completed_before = []
    for arm in arms:
        block_hash = blocks["artifacts"][arm]["file_sha256"]
        for seed in seeds:
            if _completed_run(
                output_dir,
                arm,
                seed,
                steps,
                plan_hash,
                block_hash,
                eval_hash,
            ):
                completed_before.append({"arm": arm, "seed": seed})
                continue
            if max_runs is not None and len(executed) >= max_runs:
                continue
            result = train_one(
                plan_path,
                block_manifest_path,
                eval_inputs_path,
                output_dir,
                arm,
                seed,
            )
            executed.append({"arm": arm, "seed": seed, "status": result["status"]})
            del result
            torch.cuda.empty_cache()

    remaining = []
    for arm in arms:
        block_hash = blocks["artifacts"][arm]["file_sha256"]
        for seed in seeds:
            if not _completed_run(
                output_dir,
                arm,
                seed,
                steps,
                plan_hash,
                block_hash,
                eval_hash,
            ):
                remaining.append({"arm": arm, "seed": seed})
    summary = {
        "schema_version": "redundancy-proxy-train-missing-summary-v1",
        "status": (
            "redundancy_proxy_training_complete"
            if not remaining
            else "redundancy_proxy_training_incomplete"
        ),
        "completed_before": completed_before,
        "executed": executed,
        "remaining": remaining,
    }
    save_json(output_dir / "qlora_runs" / "train_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _csv(value: str | None, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _int_csv(value: str | None, default: Iterable[int]) -> List[int]:
    if value is None:
        return [int(item) for item in default]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run redundancy proxy QLoRA.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--block-manifest", type=Path, default=DEFAULT_BLOCK_MANIFEST)
    parser.add_argument("--eval-inputs", type=Path, default=DEFAULT_EVAL_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--output", type=Path, default=DEFAULT_AUDIT)

    one = sub.add_parser("train-one")
    one.add_argument("--arm", required=True)
    one.add_argument("--seed", type=int, required=True)

    missing = sub.add_parser("train-missing")
    missing.add_argument("--arms")
    missing.add_argument("--seeds")
    missing.add_argument("--max-runs", type=int)

    args = parser.parse_args()
    if args.command == "audit":
        result = audit_runner(
            args.plan,
            args.block_manifest,
            args.eval_inputs,
            args.output,
        )
        print(json.dumps(result, indent=2))
        return 0 if not result["blockers"] else 2
    if args.command == "train-one":
        train_one(
            args.plan,
            args.block_manifest,
            args.eval_inputs,
            args.output_dir,
            args.arm,
            args.seed,
        )
        return 0

    plan = load_json(args.plan)
    arms = _csv(args.arms, plan["arms"].keys())
    seeds = _int_csv(args.seeds, plan["training_recipe"]["seeds"])
    train_missing(
        args.plan,
        args.block_manifest,
        args.eval_inputs,
        args.output_dir,
        arms,
        seeds,
        args.max_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
