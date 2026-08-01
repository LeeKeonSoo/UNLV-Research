#!/usr/bin/env python3
"""Run frozen external general-text NLL guardrail for code-domain QLoRA arms."""

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


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_HOLDOUT = (
    OUTPUT_DIR
    / "slm_update_experiments"
    / "fineweb_edu_canonical_slm_update_v1"
    / "external_guardrails"
    / "wikitext103_validation_test_guardrail.jsonl"
)
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)


class BlockDataset(Dataset):
    def __init__(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu")
        self.blocks = payload["input_ids"].to(torch.long)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        value = self.blocks[index]
        return {"input_ids": value, "labels": value.clone()}


def _parse_csv(value: str | None, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str | None, default: Iterable[int]) -> List[int]:
    if value is None:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _training_recipe(plan: Dict[str, Any]) -> Dict[str, Any]:
    if "training_recipe" in plan:
        return plan["training_recipe"]
    return plan["confirmatory_training_recipe"]


def _training_seeds(plan: Dict[str, Any]) -> List[int]:
    recipe = _training_recipe(plan)
    if "development_training_seeds" in recipe:
        return [int(seed) for seed in recipe["development_training_seeds"]]
    return [int(seed) for seed in recipe["confirmatory_training_seeds"]]


def _trained_arms(plan: Dict[str, Any]) -> List[str]:
    arms = [str(arm) for arm in plan.get("training_arms") or [] if str(arm) != "base_no_update"]
    return arms or list(TRAINED_ARMS)


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

    tokenizer = AutoTokenizer.from_pretrained(
        plan["target_model"]["tokenizer_id"],
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_eval_model(plan: Dict[str, Any], output_dir: Path, arm: str, seed: int | None, allow_download: bool) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        plan["target_model"]["model_id"],
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=not allow_download,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    if arm != "base_no_update":
        from peft import PeftModel

        if seed is None:
            raise ValueError("seed is required for adapter evaluation")
        steps = int(_training_recipe(plan)["optimizer_steps"])
        adapter_path = output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model


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


def prepare_blocks(
    plan_path: Path,
    output_dir: Path,
    holdout_path: Path,
    allow_download: bool,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    tokenizer = _load_tokenizer(plan, allow_download)
    sequence_length = int(_training_recipe(plan)["sequence_length"])
    blocks: List[torch.Tensor] = []
    buffer: List[int] = []
    consumed = 0
    for token_id in _token_stream(holdout_path, tokenizer):
        buffer.append(token_id)
        consumed += 1
        if len(buffer) == sequence_length:
            blocks.append(torch.tensor(buffer, dtype=torch.int32))
            buffer = []
    if not blocks:
        raise RuntimeError("No general-text guardrail blocks produced.")
    tensor = torch.stack(blocks)
    block_dir = output_dir / "general_text_guardrail" / "token_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    block_path = block_dir / "wikitext103_qwen3_blocks.pt"
    torch.save({"input_ids": tensor}, block_path)
    manifest = {
        "schema_version": "code-domain-general-text-guardrail-blocks-v1",
        "status": "general_text_guardrail_blocks_frozen",
        "holdout": str(holdout_path),
        "holdout_sha256": sha256_file(holdout_path),
        "path": str(block_path),
        "sha256": sha256_file(block_path),
        "blocks": int(tensor.shape[0]),
        "sequence_length": sequence_length,
        "packed_tokens": int(tensor.numel()),
        "consumed_tokens_before_packing": consumed,
        "dropped_tail_tokens": consumed - int(tensor.numel()),
        "confirmatory_outcomes_read": False,
    }
    save_json(block_dir / "wikitext103_qwen3_blocks_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def _nll_result_path(output_dir: Path, arm: str, seed: int | None) -> Path:
    name = "base_no_update" if arm == "base_no_update" else f"{arm}_seed{seed}"
    return output_dir / "general_text_guardrail" / "nll" / f"{name}.json"


def _completed_eval(output_dir: Path, arm: str, seed: int | None) -> bool:
    path = _nll_result_path(output_dir, arm, seed)
    if not path.exists():
        return False
    try:
        row = load_json(path)
    except json.JSONDecodeError:
        return False
    return row.get("status") == "general_text_nll_completed"


@torch.no_grad()
def evaluate_one(
    plan_path: Path,
    output_dir: Path,
    block_path: Path,
    arm: str,
    seed: int | None,
    allow_download: bool,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for general-text guardrail evaluation.")
    plan = load_json(plan_path)
    _set_seed(0 if seed is None else int(seed))
    started = time.time()
    model = _load_eval_model(plan, output_dir, arm, seed, allow_download)
    loader = DataLoader(BlockDataset(block_path), batch_size=1, shuffle=False)
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
        "schema_version": "code-domain-general-text-nll-result-v1",
        "status": "general_text_nll_completed",
        "arm": arm,
        "seed": seed,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else None,
        "tokens": total_tokens,
        "block_path": str(block_path),
        "block_sha256": sha256_file(block_path),
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "General-text NLL retention guardrail only; Stage C evidence, never selector objective.",
    }
    save_json(_nll_result_path(output_dir, arm, seed), result)
    print(json.dumps(result, indent=2))
    return result


def evaluate_missing(
    plan_path: Path,
    output_dir: Path,
    block_path: Path,
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
        if _completed_eval(output_dir, str(task["arm"]), task["seed"]):
            skipped.append({"arm": task["arm"], "seed": task["seed"], "status": "already_complete"})
            continue
        if max_evals is not None and len(executed) >= max_evals:
            continue
        result = evaluate_one(plan_path, output_dir, block_path, str(task["arm"]), task["seed"], allow_download)
        executed.append({"arm": result["arm"], "seed": result["seed"], "mean_nll": result["mean_nll"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "schema_version": "code-domain-general-text-evaluate-missing-summary-v1",
        "status": "general_text_evaluate_missing_completed",
        "executed": executed,
        "skipped": skipped,
        "remaining": [
            task for task in tasks if not _completed_eval(output_dir, str(task["arm"]), task["seed"])
        ],
    }
    save_json(output_dir / "general_text_guardrail" / "evaluate_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run code-domain general-text guardrail.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--allow-download", action="store_true")

    prep = sub.add_parser("prepare-blocks", parents=[common])
    prep.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)

    one = sub.add_parser("eval", parents=[common])
    one.add_argument("--blocks", type=Path, default=DEFAULT_OUTPUT_DIR / "general_text_guardrail" / "token_blocks" / "wikitext103_qwen3_blocks.pt")
    one.add_argument("--arm", required=True)
    one.add_argument("--seed", type=int)

    missing = sub.add_parser("eval-missing", parents=[common])
    missing.add_argument("--blocks", type=Path, default=DEFAULT_OUTPUT_DIR / "general_text_guardrail" / "token_blocks" / "wikitext103_qwen3_blocks.pt")
    missing.add_argument("--arms")
    missing.add_argument("--seeds")
    missing.add_argument("--max-evals", type=int)

    args = parser.parse_args()
    plan = load_json(args.plan)
    default_seeds = _training_seeds(plan)
    if args.command == "prepare-blocks":
        prepare_blocks(args.plan, args.output_dir, args.holdout, args.allow_download)
    elif args.command == "eval":
        evaluate_one(args.plan, args.output_dir, args.blocks, args.arm, args.seed, args.allow_download)
    elif args.command == "eval-missing":
        evaluate_missing(
            args.plan,
            args.output_dir,
            args.blocks,
            _parse_csv(args.arms, _trained_arms(plan)),
            _parse_int_csv(args.seeds, default_seeds),
            args.max_evals,
            args.allow_download,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
