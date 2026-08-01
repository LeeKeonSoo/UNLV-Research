#!/usr/bin/env python3
"""Run frozen target-SLM continued-pretraining and NLL evaluation arms."""

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

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
TRAINING_ARMS = (
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
)


class BlockDataset(Dataset):
    def __init__(self, blocks_path: Path) -> None:
        payload = torch.load(blocks_path, map_location="cpu")
        blocks = payload["input_ids"] if isinstance(payload, dict) else payload
        if blocks.dtype != torch.long:
            blocks = blocks.to(torch.long)
        self.blocks = blocks

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.blocks[idx]
        return {"input_ids": item, "labels": item.clone()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _visible_device_summary() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda_device_count": 0, "gpus": []}
    return {
        "device": "cuda",
        "cuda_device_count": torch.cuda.device_count(),
        "gpus": [
            {
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory": int(torch.cuda.get_device_properties(idx).total_memory),
            }
            for idx in range(torch.cuda.device_count())
        ],
    }


def _load_tokenizer(plan: Dict[str, Any], *, local_files_only: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer_id = str(((plan.get("target_model") or {}).get("tokenizer_id") or (plan.get("target_model") or {}).get("model_id") or ""))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=local_files_only, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(plan: Dict[str, Any], *, local_files_only: bool, dtype: str) -> Any:
    from transformers import AutoModelForCausalLM

    model_id = str(((plan.get("target_model") or {}).get("model_id") or ""))
    torch_dtype = torch.float32
    if dtype == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif dtype == "bf16" and torch.cuda.is_available():
        torch_dtype = torch.float16
    elif dtype == "fp16":
        # Keep fp32 master weights and use autocast+GradScaler during training.
        torch_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


def _arm_path(plan: Dict[str, Any], arm: str) -> Path:
    arms = plan.get("arm_token_counts") if isinstance(plan.get("arm_token_counts"), dict) else {}
    path_value = str((arms.get(arm) or {}).get("path") or "")
    path = Path(path_value) if path_value else Path("")
    if path_value and path.is_file():
        return path
    plan_path = Path(str(plan.get("_plan_path") or ""))
    experiment_dir = plan_path.parent if plan_path.exists() else DEFAULT_EXPERIMENT_DIR
    fallback = experiment_dir / f"{arm}.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing arm JSONL for {arm}: {path}; fallback={fallback}")


def _eos_id(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "eos_token_id", None)
    return int(value) if value is not None else None


def _token_stream_from_jsonl(
    path: Path,
    tokenizer: Any,
    *,
    max_tokens: int | None,
) -> Iterable[int]:
    emitted = 0
    eos = _eos_id(tokenizer)
    for record in iter_jsonl_records_resilient(path):
        text = str(record.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos is not None:
            ids.append(eos)
        for token_id in ids:
            if max_tokens is not None and emitted >= max_tokens:
                return
            yield int(token_id)
            emitted += 1


def build_blocks(
    *,
    jsonl_path: Path,
    output_path: Path,
    tokenizer: Any,
    sequence_length: int,
    max_tokens: int | None,
    max_sequences: int | None,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blocks: List[torch.Tensor] = []
    buffer: List[int] = []
    consumed_tokens = 0
    started = time.time()
    for token_id in _token_stream_from_jsonl(jsonl_path, tokenizer, max_tokens=max_tokens):
        buffer.append(token_id)
        consumed_tokens += 1
        while len(buffer) >= sequence_length:
            blocks.append(torch.tensor(buffer[:sequence_length], dtype=torch.int32))
            del buffer[:sequence_length]
            if max_sequences is not None and len(blocks) >= max_sequences:
                break
        if max_sequences is not None and len(blocks) >= max_sequences:
            break
    if not blocks:
        raise RuntimeError(f"No token blocks created from {jsonl_path}")
    tensor = torch.stack(blocks)
    payload = {
        "input_ids": tensor,
        "metadata": {
            "source_jsonl": str(jsonl_path),
            "sequence_length": int(sequence_length),
            "blocks": int(tensor.shape[0]),
            "tokens_in_blocks": int(tensor.numel()),
            "consumed_stream_tokens": int(consumed_tokens),
            "max_tokens": max_tokens,
            "max_sequences": max_sequences,
            "elapsed_seconds": round(time.time() - started, 3),
        },
    }
    torch.save(payload, output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    save_json(meta_path, payload["metadata"])
    return payload["metadata"]


def prepare_blocks(args: argparse.Namespace) -> Dict[str, Any]:
    plan = load_json(args.plan)
    plan["_plan_path"] = str(args.plan)
    tokenizer = _load_tokenizer(plan, local_files_only=not bool(args.allow_download))
    sequence_length = int(args.sequence_length or ((plan.get("token_budget") or {}).get("sequence_length") or 1024))
    budget_tokens = int(args.token_budget or ((plan.get("token_budget") or {}).get("all_equal_budget_arms_matched_token_budget") or 0))
    out_dir = Path(args.blocks_dir or (Path(args.plan).parent / "token_blocks"))
    manifest_path = out_dir / "block_manifest.json"
    existing_manifest = load_json(manifest_path) if manifest_path.exists() else {}
    existing_blocks = existing_manifest.get("blocks") if isinstance(existing_manifest.get("blocks"), dict) else {}
    results: Dict[str, Any] = {}
    arms = list(TRAINING_ARMS) if args.arms is None else [str(arm) for arm in args.arms]
    for arm in arms:
        results[arm] = build_blocks(
            jsonl_path=_arm_path(plan, arm),
            output_path=out_dir / f"{arm}.pt",
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            max_tokens=budget_tokens,
            max_sequences=args.max_sequences,
        )
    if args.eval_jsonl:
        eval_name = str(args.eval_name or "eval")
        results["eval"] = build_blocks(
            jsonl_path=Path(args.eval_jsonl),
            output_path=out_dir / f"{eval_name}.pt",
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            max_tokens=args.eval_token_budget,
            max_sequences=args.max_eval_sequences,
        )
        if eval_name != "eval":
            results[eval_name] = results.pop("eval")
    merged = {**existing_blocks, **results}
    save_json(manifest_path, {"schema_version": "slm-update-token-blocks-v1", "blocks": merged})
    return merged


def _configure_trainable_params(model: Any, train_mode: str, last_n_layers: int) -> Dict[str, Any]:
    for param in model.parameters():
        param.requires_grad = train_mode == "full"
    if train_mode == "lm_head":
        for name, param in model.named_parameters():
            if "lm_head" in name or "embed_tokens" in name:
                param.requires_grad = True
    elif train_mode == "last_n_layers":
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise RuntimeError("Cannot find model.layers for last_n_layers training mode")
        for layer in layers[-int(last_n_layers):]:
            for param in layer.parameters():
                param.requires_grad = True
        for name, param in model.named_parameters():
            if "lm_head" in name or "norm" in name:
                param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"train_mode": train_mode, "trainable_parameters": int(trainable), "total_parameters": int(total)}


def _unwrap_model(model: Any) -> Any:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def train_arm(args: argparse.Namespace) -> Dict[str, Any]:
    _set_seed(int(args.seed))
    plan = load_json(args.plan)
    plan["_plan_path"] = str(args.plan)
    blocks_dir = Path(args.blocks_dir or (Path(args.plan).parent / "token_blocks"))
    train_path = blocks_dir / f"{args.arm}.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing token blocks: {train_path}. Run prepare-blocks first.")
    model = _load_model(plan, local_files_only=not bool(args.allow_download), dtype=str(args.dtype))
    trainable = _configure_trainable_params(model, str(args.train_mode), int(args.last_n_layers))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
    dataset = BlockDataset(train_path)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    model.train()
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    max_steps = int(args.max_steps or math.ceil(len(loader) / grad_accum))
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and str(args.dtype) == "fp16"))
    losses: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    started = time.time()
    while step < max_steps:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            autocast_dtype = torch.bfloat16 if str(args.dtype) == "bf16" and torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda" and str(args.dtype) in {"bf16", "fp16"})):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss.mean() / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            losses.append(float(loss.detach().cpu()) * grad_accum)
            micro_step += 1
            if micro_step % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.max_grad_norm))
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % int(args.logging_steps) == 0 or step == 1:
                    print(json.dumps({"step": step, "loss": round(sum(losses[-grad_accum:]) / max(1, min(len(losses), grad_accum)), 6)}))
                if step >= max_steps:
                    break
    out_dir = Path(args.output_dir or (Path(args.plan).parent / "model_runs" / f"{args.arm}_seed{args.seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    _unwrap_model(model).save_pretrained(out_dir)
    tokenizer = _load_tokenizer(plan, local_files_only=not bool(args.allow_download))
    tokenizer.save_pretrained(out_dir)
    result = {
        "schema_version": "slm-update-train-run-v1",
        "arm": args.arm,
        "seed": int(args.seed),
        "plan": str(args.plan),
        "train_blocks": str(train_path),
        "device_summary": _visible_device_summary(),
        "trainable": trainable,
        "steps": int(step),
        "micro_steps": int(micro_step),
        "mean_loss": float(sum(losses) / len(losses)) if losses else None,
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(out_dir),
    }
    save_json(out_dir / "train_result.json", result)
    print(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def evaluate_model(args: argparse.Namespace) -> Dict[str, Any]:
    plan = load_json(args.plan)
    blocks_dir = Path(args.blocks_dir or (Path(args.plan).parent / "token_blocks"))
    eval_path = Path(args.eval_blocks or (blocks_dir / "eval.pt"))
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval token blocks: {eval_path}")
    from transformers import AutoModelForCausalLM

    model_source = str(args.model_path or ((plan.get("target_model") or {}).get("model_id") or ""))
    model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=not bool(args.allow_download))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1 and not args.single_gpu:
        model = torch.nn.DataParallel(model)
    model.eval()
    loader = DataLoader(BlockDataset(eval_path), batch_size=int(args.batch_size), shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss.mean()
        tokens = int(input_ids.numel())
        total_loss += float(loss.detach().cpu()) * tokens
        total_tokens += tokens
        if args.max_eval_batches and total_tokens >= int(args.max_eval_batches) * int(args.batch_size) * int(input_ids.shape[-1]):
            break
    mean_nll = total_loss / max(1, total_tokens)
    result = {
        "schema_version": "slm-update-eval-result-v1",
        "model_path": model_source,
        "eval_blocks": str(eval_path),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else None,
        "tokens": int(total_tokens),
        "device_summary": _visible_device_summary(),
    }
    output = Path(args.output or (Path(args.plan).parent / "eval_results" / f"{Path(model_source).name}_eval.json"))
    save_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen SLM update training/eval.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_plan = argparse.ArgumentParser(add_help=False)
    common_plan.add_argument("--plan", type=Path, default=DEFAULT_EXPERIMENT_DIR / "frozen_training_plan.json")
    common_plan.add_argument("--allow-download", action="store_true")

    p_blocks = sub.add_parser("prepare-blocks", parents=[common_plan])
    p_blocks.add_argument("--arms", nargs="*")
    p_blocks.add_argument("--blocks-dir", type=Path)
    p_blocks.add_argument("--sequence-length", type=int)
    p_blocks.add_argument("--token-budget", type=int)
    p_blocks.add_argument("--max-sequences", type=int)
    p_blocks.add_argument("--eval-jsonl", type=Path)
    p_blocks.add_argument("--eval-name")
    p_blocks.add_argument("--eval-token-budget", type=int)
    p_blocks.add_argument("--max-eval-sequences", type=int)
    p_blocks.set_defaults(func=prepare_blocks)

    p_train = sub.add_parser("train", parents=[common_plan])
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--blocks-dir", type=Path)
    p_train.add_argument("--output-dir", type=Path)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p_train.add_argument("--learning-rate", type=float, default=5e-5)
    p_train.add_argument("--weight-decay", type=float, default=0.1)
    p_train.add_argument("--max-grad-norm", type=float, default=1.0)
    p_train.add_argument("--max-steps", type=int)
    p_train.add_argument("--logging-steps", type=int, default=25)
    p_train.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p_train.add_argument("--train-mode", choices=("full", "lm_head", "last_n_layers"), default="full")
    p_train.add_argument("--last-n-layers", type=int, default=2)
    p_train.add_argument("--single-gpu", action="store_true")
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=train_arm)

    p_eval = sub.add_parser("eval", parents=[common_plan])
    p_eval.add_argument("--model-path", type=Path)
    p_eval.add_argument("--blocks-dir", type=Path)
    p_eval.add_argument("--eval-blocks", type=Path)
    p_eval.add_argument("--output", type=Path)
    p_eval.add_argument("--batch-size", type=int, default=1)
    p_eval.add_argument("--max-eval-batches", type=int)
    p_eval.add_argument("--single-gpu", action="store_true")
    p_eval.add_argument("--cpu", action="store_true")
    p_eval.set_defaults(func=evaluate_model)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
