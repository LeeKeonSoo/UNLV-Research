#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Final, TypedDict

import torch


ROOT: Final = Path(__file__).resolve().parents[1]
PROTOCOL_PATH: Final = ROOT / "protocols" / "code_evalplus_natural_3arm_qwen3_4b_v1.json"
SPLIT_PATH: Final = ROOT / "outputs" / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
RUN_ROOT: Final = Path("D:/UNLV-Research/code_5m_corpus_v2/final_replay_v1/evalplus_natural_v1/runs")
DATASETS: Final = ("HumanEval+", "MBPP+")
GENERATION_SCOPES: Final = ("development", "official_full")


class SplitRecord(TypedDict):
    dataset: str
    task_id: str
    assigned_split: str


def adapter_directory(run_root: Path, arm: str, seed: int, steps: int) -> Path:
    return run_root / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def task_ids_for_split(records: list[dict[str, str]], dataset: str, split: str) -> list[str]:
    return sorted(
        record["task_id"]
        for record in records
        if record["dataset"] == dataset and record["assigned_split"] == split
    )


def task_ids_for_scope(
    records: list[dict[str, str]], tasks: dict[str, dict[str, object]], dataset: str, scope: str
) -> list[str]:
    if scope == "official_full":
        return sorted(tasks)
    if scope == "development":
        return task_ids_for_split(records, dataset, "development")
    raise ValueError(f"Unsupported EvalPlus generation scope: {scope}")


def trim_completion(text: str) -> str:
    content = text
    if "```" in content:
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else parts[0]
        content = content.lstrip()
        if content.startswith("python"):
            content = content[len("python") :].lstrip()
    for marker in ("\n```", "\nif __name__ ==", "\n# Task", "\n###", "\n\n\n"):
        position = content.find(marker)
        if position > 0:
            content = content[:position]
    return content.rstrip() + "\n"


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return raw


def load_tasks(dataset: str) -> dict[str, dict[str, object]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "HumanEval+":
        return get_human_eval_plus()
    if dataset == "MBPP+":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


def load_model(protocol: dict[str, object], arm: str, seed: int | None) -> tuple[object, object]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    target = protocol["target_model"]
    recipe = protocol["training_recipe"]
    if not isinstance(target, dict) or not isinstance(recipe, dict):
        raise TypeError("Invalid frozen EvalPlus protocol")
    snapshot = target["snapshot_path"]
    if not isinstance(snapshot, str):
        raise TypeError("Frozen model snapshot is missing")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    if arm != "base_no_update":
        if seed is None:
            raise ValueError("A seed is required for an adapter arm")
        steps_by_arm = recipe["optimizer_steps_by_arm"]
        if not isinstance(steps_by_arm, dict):
            raise TypeError("Frozen optimizer steps are missing")
        steps = steps_by_arm.get(arm)
        if not isinstance(steps, int):
            raise ValueError(f"Unknown adapter arm: {arm}")
        checkpoint = adapter_directory(RUN_ROOT, arm, seed, steps)
        if not (checkpoint / "adapter_config.json").exists():
            raise FileNotFoundError(f"Adapter is not complete: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint, local_files_only=True)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(arm: str, seed: int | None, batch_size: int, max_new_tokens: int, scope: str) -> list[Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("EvalPlus generation requires CUDA")
    protocol = load_json(PROTOCOL_PATH)
    split = load_json(SPLIT_PATH)
    raw_records = split.get("records")
    if not isinstance(raw_records, list):
        raise TypeError("Frozen EvalPlus split records are missing")
    records = [record for record in raw_records if isinstance(record, dict)]
    random.seed(0 if seed is None else seed)
    torch.manual_seed(0 if seed is None else seed)
    torch.cuda.manual_seed_all(0 if seed is None else seed)
    model, tokenizer = load_model(protocol, arm, seed)
    samples_dir = RUN_ROOT / ("evalplus_official" if scope == "official_full" else "evalplus_guardrail") / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    completed: list[Path] = []
    for dataset in DATASETS:
        tasks = load_tasks(dataset)
        task_ids = task_ids_for_scope(records, tasks, dataset, scope)
        slug = "humaneval" if dataset == "HumanEval+" else "mbpp"
        suffix = "base" if seed is None else f"seed{seed}"
        output = samples_dir / f"{slug}_{arm}_{suffix}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for start in range(0, len(task_ids), batch_size):
                batch_ids = task_ids[start : start + batch_size]
                prompts = [str(tasks[task_id]["prompt"]) for task_id in batch_ids]
                encoded = tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(0)
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_width = encoded["input_ids"].shape[1]
                for task_id, sequence in zip(batch_ids, generated):
                    completion = trim_completion(tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True))
                    handle.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
        completed.append(output)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen EvalPlus samples from the active QLoRA runs.")
    parser.add_argument("--arm", required=True, choices=("base_no_update", "raw_safe_natural", "curated_natural"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--scope", choices=GENERATION_SCOPES, default="official_full")
    args = parser.parse_args()
    if args.arm == "base_no_update" and args.seed is not None:
        parser.error("base_no_update does not accept --seed")
    if args.arm != "base_no_update" and args.seed is None:
        parser.error("adapter arms require --seed")
    for output in generate(args.arm, args.seed, args.batch_size, args.max_new_tokens, args.scope):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
