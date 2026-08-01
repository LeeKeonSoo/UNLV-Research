#!/usr/bin/env python3
"""Run and evaluate frozen code-domain QLoRA development or confirmatory plans."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
PROJECT_DIR = Path(__file__).resolve().parents[2]
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)


def _training_recipe(plan: Dict[str, Any]) -> Dict[str, Any]:
    if "training_recipe" in plan:
        return plan["training_recipe"]
    return plan["confirmatory_training_recipe"]


def _optimizer_steps(plan: Dict[str, Any], arm: str | None = None) -> int:
    recipe = _training_recipe(plan)
    by_arm = recipe.get("optimizer_steps_by_arm")
    if arm is not None and isinstance(by_arm, dict) and arm in by_arm:
        return int(by_arm[arm])
    return int(recipe["optimizer_steps"])


def _evaluation_optimizer_steps(plan: Dict[str, Any], arm: str) -> int:
    return 0 if arm == "base_no_update" else _optimizer_steps(plan, arm)


def _training_seeds(plan: Dict[str, Any]) -> List[int]:
    recipe = _training_recipe(plan)
    if "development_training_seeds" in recipe:
        return [int(seed) for seed in recipe["development_training_seeds"]]
    return [int(seed) for seed in recipe["confirmatory_training_seeds"]]


def _trained_arms(plan: Dict[str, Any]) -> List[str]:
    arms = [str(arm) for arm in plan.get("training_arms") or [] if str(arm) != "base_no_update"]
    return arms or list(TRAINED_ARMS)


def _stage_label(plan: Dict[str, Any]) -> str:
    return "confirmatory" if "confirmatory_training_recipe" in plan else "development"


def _qlora_completed_status(plan: Dict[str, Any]) -> str:
    return f"{_stage_label(plan)}_qlora_completed"


def _eval_blocks_name(plan: Dict[str, Any]) -> str:
    if _stage_label(plan) == "confirmatory":
        return "confirmatory_code_nll_heldout.pt"
    return "development_code_nll_heldout.pt"


def _heldout_jsonl_path(plan: Dict[str, Any]) -> Path:
    heldout = plan["heldout_nll"]
    if "frozen_heldout" in heldout:
        return _resolve(str(heldout["frozen_heldout"]["path"]))
    return _resolve("outputs/code_domain_development_qwen3_4b_v1/heldouts/development_code_nll_heldout.jsonl")


class BlockDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = (
            load_file(path, device="cpu")
            if path.suffix == ".safetensors"
            else torch.load(path, map_location="cpu")
        )
        self.blocks = payload["input_ids"].to(torch.long)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        value = self.blocks[index]
        return {"input_ids": value, "labels": value.clone()}


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _load_tokenizer(plan: Dict[str, Any], allow_download: bool) -> Any:
    from transformers import AutoTokenizer

    target = plan["target_model"].get("snapshot_path") or plan["target_model"]["tokenizer_id"]
    tokenizer = AutoTokenizer.from_pretrained(
        target,
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_qlora_model(plan: Dict[str, Any], allow_download: bool) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    recipe = _training_recipe(plan)
    adapter = recipe["adapter"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    target = plan["target_model"].get("snapshot_path") or plan["target_model"]["model_id"]
    model = AutoModelForCausalLM.from_pretrained(
        target,
        revision=plan["target_model"].get("revision", "main"),
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


def _load_eval_model(plan: Dict[str, Any], allow_download: bool, adapter_path: Path | None) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    target = plan["target_model"].get("snapshot_path") or plan["target_model"]["model_id"]
    model = AutoModelForCausalLM.from_pretrained(
        target,
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model


def _run_dir(output_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def _completed_run(output_dir: Path, arm: str, seed: int, steps: int) -> bool:
    path = _run_dir(output_dir, arm, seed, steps) / "run_result.json"
    if not path.exists():
        return False
    try:
        result = load_json(path)
    except json.JSONDecodeError:
        return False
    return result.get("status") in {
        "development_qlora_completed",
        "confirmatory_qlora_completed",
    } and int(result.get("optimizer_steps") or 0) == steps


def train_one(
    plan_path: Path,
    output_dir: Path,
    blocks_dir: Path,
    arm: str,
    seed: int,
    allow_download: bool,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    recipe = _training_recipe(plan)
    stage = _stage_label(plan)
    steps = _optimizer_steps(plan, arm)
    trained_arms = _trained_arms(plan)
    if arm not in trained_arms:
        raise ValueError(f"{stage.title()} training arm must be one of {trained_arms}: {arm}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required for QLoRA {stage} training.")
    _set_seed(seed)
    started = time.time()
    blocks_path = blocks_dir / f"{arm}.pt"
    if not blocks_path.exists():
        safetensors_path = blocks_dir / f"{arm}.safetensors"
        if safetensors_path.exists():
            blocks_path = safetensors_path
    if not blocks_path.exists():
        raise FileNotFoundError(f"Missing frozen token blocks: {blocks_path}")
    model = _load_qlora_model(plan, allow_download=allow_download)
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
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: List[float] = []
    optimizer_steps = 0
    micro_steps = 0
    while optimizer_steps < steps:
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
                print(json.dumps({"arm": arm, "seed": seed, "optimizer_step": optimizer_steps, "loss": losses[-1]}))
                if optimizer_steps >= steps:
                    break
    run_dir = _run_dir(output_dir, arm, seed, steps)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run_dir)
    result = {
        "schema_version": f"code-domain-{stage}-qlora-run-v1",
        "status": _qlora_completed_status(plan),
        "stage": stage,
        "arm": arm,
        "seed": int(seed),
        "optimizer_steps": optimizer_steps,
        "micro_steps": micro_steps,
        "mean_microbatch_loss": sum(losses) / max(1, len(losses)),
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "plan_sha256": sha256_file(plan_path),
        "train_blocks": str(blocks_path),
        "train_blocks_sha256": sha256_file(blocks_path),
        "device_summary": _device_summary(),
        "cuda_peak_memory_allocated": int(torch.cuda.max_memory_allocated(0)),
        "elapsed_seconds": round(time.time() - started, 3),
        "output_dir": str(run_dir),
        "utility_scope": plan["utility_scope"],
        "claim_boundary": f"{stage.title()} QLoRA training artifact only; no Utility or release claim.",
    }
    save_json(run_dir / "run_result.json", result)
    print(json.dumps(result, indent=2))
    return result


def train_missing(
    plan_path: Path,
    output_dir: Path,
    blocks_dir: Path,
    arms: List[str],
    seeds: List[int],
    max_runs: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    stage = _stage_label(plan)
    completed = []
    executed = []
    for arm in arms:
        for seed in seeds:
            steps = _optimizer_steps(plan, arm)
            if _completed_run(output_dir, arm, seed, steps):
                completed.append({"arm": arm, "seed": seed, "status": "already_complete"})
                continue
            if max_runs is not None and len(executed) >= max_runs:
                continue
            result = train_one(plan_path, output_dir, blocks_dir, arm, seed, allow_download)
            executed.append({"arm": arm, "seed": seed, "status": result["status"]})
            torch.cuda.empty_cache()
    summary = {
        "schema_version": f"code-domain-{stage}-train-missing-summary-v1",
        "stage": stage,
        "plan": str(plan_path),
        "steps_by_arm": {arm: _optimizer_steps(plan, arm) for arm in arms},
        "completed_before": completed,
        "executed": executed,
        "remaining": [
            {"arm": arm, "seed": seed}
            for arm in arms
            for seed in seeds
            if not _completed_run(output_dir, arm, seed, steps)
        ],
    }
    save_json(output_dir / "qlora_runs" / "train_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _token_stream(path: Path, tokenizer: Any) -> Iterable[int]:
    eos = tokenizer.eos_token_id
    for row in iter_jsonl_records_resilient(path):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos is not None:
            ids.append(int(eos))
        for token_id in ids:
            yield int(token_id)


def prepare_eval_blocks(plan_path: Path, output_dir: Path, allow_download: bool) -> Dict[str, Any]:
    plan = load_json(plan_path)
    tokenizer = _load_tokenizer(plan, allow_download=allow_download)
    stage = _stage_label(plan)
    heldout = _heldout_jsonl_path(plan)
    sequence_length = int(_training_recipe(plan)["sequence_length"])
    blocks: List[torch.Tensor] = []
    buffer: List[int] = []
    consumed = 0
    for token_id in _token_stream(heldout, tokenizer):
        buffer.append(token_id)
        consumed += 1
        if len(buffer) == sequence_length:
            blocks.append(torch.tensor(buffer, dtype=torch.int32))
            buffer = []
    if not blocks:
        raise RuntimeError(f"No eval blocks produced from {stage} heldout.")
    tensor = torch.stack(blocks)
    output = output_dir / "eval_blocks" / _eval_blocks_name(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": tensor}, output)
    report = {
        "schema_version": f"code-domain-{stage}-eval-blocks-v1",
        "status": f"{stage}_eval_blocks_frozen",
        "stage": stage,
        "heldout_jsonl": str(heldout),
        "heldout_sha256": sha256_file(heldout),
        "path": str(output),
        "sha256": sha256_file(output),
        "blocks": int(tensor.shape[0]),
        "sequence_length": sequence_length,
        "packed_tokens": int(tensor.numel()),
        "consumed_tokens_before_packing": consumed,
        "dropped_tail_tokens": consumed - int(tensor.numel()),
        "confirmatory_outcomes_read": False,
    }
    save_json(output.parent / "eval_block_manifest.json", report)
    print(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def evaluate_one(
    plan_path: Path,
    output_dir: Path,
    eval_blocks_path: Path,
    arm: str,
    seed: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    stage = _stage_label(plan)
    steps = _evaluation_optimizer_steps(plan, arm)
    adapter_path = None if arm == "base_no_update" else _run_dir(output_dir, arm, int(seed), steps)
    if adapter_path is not None and not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter for evaluation: {adapter_path}")
    started = time.time()
    model = _load_eval_model(plan, allow_download=allow_download, adapter_path=adapter_path)
    loader = DataLoader(BlockDataset(eval_blocks_path), batch_size=1, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(0) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        tokens = int(batch["input_ids"].numel())
        total_loss += float(loss.detach().cpu()) * tokens
        total_tokens += tokens
    mean_nll = total_loss / max(1, total_tokens)
    result = {
        "schema_version": f"code-domain-{stage}-heldout-nll-result-v1",
        "status": "heldout_nll_completed",
        "stage": stage,
        "arm": arm,
        "seed": seed,
        "optimizer_steps": 0 if arm == "base_no_update" else steps,
        "eval_blocks": str(eval_blocks_path),
        "eval_blocks_sha256": sha256_file(eval_blocks_path),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else None,
        "tokens": total_tokens,
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "utility_scope": plan["utility_scope"],
        "claim_boundary": f"{stage.title()} heldout NLL only; no release claim.",
    }
    name = "base_no_update" if arm == "base_no_update" else f"{arm}_seed{seed}"
    output = output_dir / "heldout_nll" / f"{name}.json"
    save_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def evaluate_missing(
    plan_path: Path,
    output_dir: Path,
    eval_blocks_path: Path,
    arms: List[str],
    seeds: List[int],
    max_evals: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = [{"arm": "base_no_update", "seed": None}]
    tasks.extend({"arm": arm, "seed": seed} for arm in arms for seed in seeds)
    executed = []
    skipped = []
    for task in tasks:
        name = "base_no_update" if task["arm"] == "base_no_update" else f"{task['arm']}_seed{task['seed']}"
        output = output_dir / "heldout_nll" / f"{name}.json"
        if output.exists():
            try:
                row = load_json(output)
                if row.get("status") == "heldout_nll_completed":
                    skipped.append({"arm": task["arm"], "seed": task["seed"], "status": "already_complete"})
                    continue
            except json.JSONDecodeError:
                pass
        if max_evals is not None and len(executed) >= max_evals:
            continue
        result = evaluate_one(
            plan_path,
            output_dir,
            eval_blocks_path,
            str(task["arm"]),
            task["seed"],
            allow_download=allow_download,
        )
        executed.append({"arm": result["arm"], "seed": result["seed"], "mean_nll": result["mean_nll"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "schema_version": f"code-domain-{_stage_label(load_json(plan_path))}-evaluate-missing-summary-v1",
        "executed": executed,
        "skipped": skipped,
        "remaining": [
            task for task in tasks
            if not (output_dir / "heldout_nll" / ("base_no_update.json" if task["arm"] == "base_no_update" else f"{task['arm']}_seed{task['seed']}.json")).exists()
        ],
    }
    save_json(output_dir / "heldout_nll" / "evaluate_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def _parse_csv(value: str | None, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str | None, default: Iterable[int]) -> List[int]:
    if value is None:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen code-domain Qwen3-4B QLoRA work.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--allow-download", action="store_true")

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--blocks-dir", type=Path, default=OUTPUT_DIR / "code_domain_qlora_smoke_qwen3_4b_v1" / "token_blocks")
    p_train.add_argument("--arm", required=True)
    p_train.add_argument("--seed", type=int, required=True)

    p_train_missing = sub.add_parser("train-missing", parents=[common])
    p_train_missing.add_argument("--blocks-dir", type=Path, default=OUTPUT_DIR / "code_domain_qlora_smoke_qwen3_4b_v1" / "token_blocks")
    p_train_missing.add_argument("--arms")
    p_train_missing.add_argument("--seeds")
    p_train_missing.add_argument("--max-runs", type=int)

    p_eval_blocks = sub.add_parser("prepare-eval-blocks", parents=[common])

    p_eval = sub.add_parser("eval", parents=[common])
    p_eval.add_argument("--eval-blocks", type=Path)
    p_eval.add_argument("--arm", required=True)
    p_eval.add_argument("--seed", type=int)

    p_eval_missing = sub.add_parser("eval-missing", parents=[common])
    p_eval_missing.add_argument("--eval-blocks", type=Path)
    p_eval_missing.add_argument("--arms")
    p_eval_missing.add_argument("--seeds")
    p_eval_missing.add_argument("--max-evals", type=int)

    args = parser.parse_args()
    plan = load_json(args.plan)
    default_seeds = _training_seeds(plan)
    default_arms = _trained_arms(plan)
    if getattr(args, "eval_blocks", None) is None:
        args.eval_blocks = args.output_dir / "eval_blocks" / _eval_blocks_name(plan)
    if args.command == "train":
        train_one(args.plan, args.output_dir, args.blocks_dir, args.arm, args.seed, args.allow_download)
    elif args.command == "train-missing":
        train_missing(
            args.plan,
            args.output_dir,
            args.blocks_dir,
            _parse_csv(args.arms, default_arms),
            _parse_int_csv(args.seeds, default_seeds),
            args.max_runs,
            args.allow_download,
        )
    elif args.command == "prepare-eval-blocks":
        prepare_eval_blocks(args.plan, args.output_dir, args.allow_download)
    elif args.command == "eval":
        evaluate_one(args.plan, args.output_dir, args.eval_blocks, args.arm, args.seed, args.allow_download)
    elif args.command == "eval-missing":
        evaluate_missing(
            args.plan,
            args.output_dir,
            args.eval_blocks,
            _parse_csv(args.arms, default_arms),
            _parse_int_csv(args.seeds, default_seeds),
            args.max_evals,
            args.allow_download,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
