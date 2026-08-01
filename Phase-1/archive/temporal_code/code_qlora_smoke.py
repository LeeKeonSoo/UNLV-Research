#!/usr/bin/env python3
"""Run Qwen3-4B QLoRA smoke on frozen code-domain equal-token arms."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


DEFAULT_CONFIG = Path("configs") / "code_domain_qlora_smoke_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_qlora_smoke_qwen3_4b_v1"
PROJECT_DIR = Path(__file__).resolve().parents[2]


class BlockDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
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


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _load_tokenizer(config: Dict[str, Any], allow_download: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["target_model"]["tokenizer_id"],
        revision=config["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _token_stream(path: Path, tokenizer: Any, token_cap: int) -> Iterable[int]:
    emitted = 0
    eos = tokenizer.eos_token_id
    for row in iter_jsonl_records_resilient(path):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos is not None:
            ids.append(int(eos))
        for token_id in ids:
            if emitted >= token_cap:
                return
            yield int(token_id)
            emitted += 1


def prepare_blocks(config_path: Path, output_dir: Path, allow_download: bool) -> Dict[str, Any]:
    config = load_json(config_path)
    arms_dir = _resolve(config["input"]["arms_dir"])
    arms_report_path = _resolve(config["input"]["arms_report"])
    arms_report = load_json(arms_report_path)
    token_cap = int(arms_report["summary"]["training_token_budget_cap"])
    sequence_length = int(config["training_recipe"]["sequence_length"])
    tokenizer = _load_tokenizer(config, allow_download=allow_download)
    blocks_dir = output_dir / "token_blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in config["arms"]:
        source = arms_dir / f"{arm}.jsonl"
        buffer: List[int] = []
        blocks: List[torch.Tensor] = []
        consumed = 0
        for token_id in _token_stream(source, tokenizer, token_cap):
            buffer.append(token_id)
            consumed += 1
            if len(buffer) == sequence_length:
                blocks.append(torch.tensor(buffer, dtype=torch.int32))
                buffer = []
        if not blocks:
            raise RuntimeError(f"No complete blocks for {arm}: consumed_tokens={consumed}")
        tensor = torch.stack(blocks)
        output = blocks_dir / f"{arm}.pt"
        torch.save({"input_ids": tensor}, output)
        results[arm] = {
            "source_jsonl": str(source),
            "source_sha256": sha256_file(source),
            "path": str(output),
            "blocks": int(tensor.shape[0]),
            "sequence_length": sequence_length,
            "packed_tokens": int(tensor.numel()),
            "consumed_tokens_before_packing": int(consumed),
            "training_token_budget_cap": token_cap,
            "dropped_tail_tokens": int(consumed - tensor.numel()),
        }
    packed = {row["packed_tokens"] for row in results.values()}
    if len(packed) != 1:
        raise RuntimeError(f"Packed token budgets differ: {results}")
    report = {
        "schema_version": "code-domain-qlora-smoke-blocks-v1",
        "status": "frozen_equal_packed_token_blocks",
        "config_sha256": sha256_file(config_path),
        "arms_report_sha256": sha256_file(arms_report_path),
        "training_token_budget_cap": token_cap,
        "common_packed_token_budget": next(iter(packed)),
        "blocks": results,
        "utility_scope": config["utility_scope"],
        "claim_boundary": "Token-block construction only; no target-model training or Utility claim.",
    }
    save_json(blocks_dir / "block_manifest.json", report)
    print(json.dumps(report["blocks"], indent=2))
    return report


def _device_summary() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_device_count": 0, "gpus": []}
    return {
        "device": "cuda",
        "cuda_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "gpus": [
            {
                "visible_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def _load_qlora_model(config: Dict[str, Any], allow_download: bool) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    recipe = config["training_recipe"]
    adapter = recipe["adapter"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["target_model"]["model_id"],
        revision=config["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
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


def train_arm(
    config_path: Path,
    output_dir: Path,
    arm: str,
    max_steps: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    config = load_json(config_path)
    recipe = config["training_recipe"]
    if arm not in config["arms"]:
        raise ValueError(f"Unknown arm {arm}; expected one of {config['arms']}")
    _set_seed(int(recipe["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA smoke.")
    started = time.time()
    blocks_path = output_dir / "token_blocks" / f"{arm}.pt"
    if not blocks_path.exists():
        raise FileNotFoundError(f"Missing token blocks: {blocks_path}")
    model = _load_qlora_model(config, allow_download=allow_download)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    loader = DataLoader(
        BlockDataset(blocks_path),
        batch_size=int(recipe["micro_batch_size"]),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    grad_accum = int(recipe["gradient_accumulation_steps"])
    target_steps = int(max_steps or recipe["smoke_optimizer_steps"])
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: List[float] = []
    optimizer_steps = 0
    micro_steps = 0
    while optimizer_steps < target_steps:
        for batch in loader:
            batch = {key: value.to(0) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            losses.append(float(loss.detach().cpu()) * grad_accum)
            micro_steps += 1
            if micro_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    float(recipe["max_grad_norm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                print(json.dumps({"arm": arm, "optimizer_step": optimizer_steps, "loss": losses[-1]}))
                if optimizer_steps >= target_steps:
                    break
    run_dir = output_dir / "qlora_runs" / f"{arm}_seed{recipe['seed']}_steps{target_steps}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir)
    result = {
        "schema_version": "code-domain-qlora-smoke-run-v1",
        "status": "qlora_smoke_completed",
        "arm": arm,
        "seed": int(recipe["seed"]),
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "mean_microbatch_loss": sum(losses) / max(1, len(losses)),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "device_summary": _device_summary(),
        "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(run_dir),
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(run_dir / "run_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run code-domain Qwen3-4B QLoRA smoke.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_blocks = sub.add_parser("prepare-blocks")
    p_blocks.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_blocks.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_blocks.add_argument("--allow-download", action="store_true")
    p_train = sub.add_parser("train")
    p_train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p_train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-blocks":
        prepare_blocks(args.config, args.output_dir, allow_download=bool(args.allow_download))
        return 0
    train_arm(args.config, args.output_dir, args.arm, args.max_steps, allow_download=bool(args.allow_download))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
