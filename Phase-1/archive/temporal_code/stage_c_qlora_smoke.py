#!/usr/bin/env python3
"""Run the frozen Qwen3-4B temporal-code QLoRA feasibility smoke."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json


DEFAULT_CONTRACT = Path("configs") / "temporal_code_stage_c_smoke_qwen3_4b_v1.json"
DEFAULT_ARMS_DIR = OUTPUT_DIR / "temporal_code_stage_c_smoke_qwen3_4b_v1"
DEFAULT_BLOCKS_DIR = DEFAULT_ARMS_DIR / "token_blocks"
DEFAULT_RUNS_DIR = DEFAULT_ARMS_DIR / "qlora_runs"
ARMS = ("curated_equal_token", "stageA_random_equal_token", "raw_random_equal_token")


class Blocks(Dataset):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.blocks = payload["input_ids"].to(torch.long)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        value = self.blocks[index]
        return {"input_ids": value, "labels": value.clone()}


def _token_stream(path: Path, tokenizer: Any) -> Iterable[int]:
    eos = tokenizer.eos_token_id
    for row in iter_jsonl_records_resilient(path):
        ids = tokenizer(str(row.get("text") or ""), add_special_tokens=False).input_ids
        for token_id in ids:
            yield int(token_id)
        if eos is not None:
            yield int(eos)


def prepare_blocks(contract_path: Path, arms_dir: Path, blocks_dir: Path) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    contract = load_json(contract_path)
    tokenizer = AutoTokenizer.from_pretrained(contract["target_model"]["tokenizer_id"], local_files_only=True)
    sequence_length = int(contract["training_recipe"]["sequence_length"])
    results = {}
    for arm in ARMS:
        buffer: List[int] = []
        blocks: List[torch.Tensor] = []
        for token_id in _token_stream(arms_dir / f"{arm}.jsonl", tokenizer):
            buffer.append(token_id)
            if len(buffer) == sequence_length:
                blocks.append(torch.tensor(buffer, dtype=torch.int32))
                buffer = []
        if not blocks:
            raise RuntimeError(f"No complete blocks for {arm}")
        tensor = torch.stack(blocks)
        output = blocks_dir / f"{arm}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_ids": tensor}, output)
        results[arm] = {
            "path": str(output),
            "blocks": int(tensor.shape[0]),
            "sequence_length": sequence_length,
            "packed_tokens": int(tensor.numel()),
        }
    if len({row["packed_tokens"] for row in results.values()}) != 1:
        raise RuntimeError(f"Packed token budgets differ: {results}")
    report = {
        "schema_version": "temporal-code-stage-c-smoke-blocks-v1",
        "status": "frozen_equal_packed_token_blocks",
        "blocks": results,
        "common_packed_token_budget": next(iter(results.values()))["packed_tokens"],
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(blocks_dir / "block_manifest.json", report)
    return report


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_qlora_model(contract: Dict[str, Any], allow_download: bool) -> Any:
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    recipe = contract["training_recipe"]
    adapter = recipe["adapter"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        contract["target_model"]["model_id"],
        revision=contract["target_model"].get("revision", "main"),
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


def train(
    contract_path: Path,
    blocks_dir: Path,
    runs_dir: Path,
    arm: str,
    *,
    max_steps: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    recipe = contract["training_recipe"]
    seed = int(recipe["smoke_seed"])
    _set_seed(seed)
    started = time.time()
    model = _load_qlora_model(contract, allow_download)
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    loader = DataLoader(
        Blocks(blocks_dir / f"{arm}.pt"),
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
    losses = []
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
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                print(json.dumps({"arm": arm, "optimizer_step": optimizer_steps, "loss": losses[-1]}))
                if optimizer_steps >= target_steps:
                    break
    output_dir = runs_dir / f"{arm}_seed{seed}_steps{target_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    result = {
        "schema_version": "temporal-code-stage-c-qlora-smoke-run-v1",
        "status": "qlora_smoke_completed",
        "arm": arm,
        "seed": seed,
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "mean_microbatch_loss": sum(losses) / len(losses),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(output_dir),
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "QLoRA execution feasibility only; no Utility or training-benefit claim.",
    }
    save_json(output_dir / "run_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run temporal-code Qwen3-4B QLoRA smoke.")
    sub = parser.add_subparsers(dest="command", required=True)
    blocks = sub.add_parser("prepare-blocks")
    blocks.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    blocks.add_argument("--arms-dir", type=Path, default=DEFAULT_ARMS_DIR)
    blocks.add_argument("--blocks-dir", type=Path, default=DEFAULT_BLOCKS_DIR)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    train_parser.add_argument("--blocks-dir", type=Path, default=DEFAULT_BLOCKS_DIR)
    train_parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-blocks":
        print(prepare_blocks(args.contract, args.arms_dir, args.blocks_dir))
        return 0
    train(args.contract, args.blocks_dir, args.runs_dir, args.arm, max_steps=args.max_steps, allow_download=args.allow_download)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
