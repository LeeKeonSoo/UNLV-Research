#!/usr/bin/env python3
"""Generate deterministic EvalPlus development samples for code-domain QLoRA arms."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_SPLIT = OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)
DATASET_MAP = {
    "HumanEval+": ("humaneval", "get_human_eval_plus"),
    "MBPP+": ("mbpp", "get_mbpp_plus"),
}


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


def _optimizer_steps(plan: Dict[str, Any], arm: str | None = None) -> int:
    recipe = _training_recipe(plan)
    by_arm = recipe.get("optimizer_steps_by_arm")
    if arm is not None and isinstance(by_arm, dict) and arm in by_arm:
        return int(by_arm[arm])
    return int(recipe["optimizer_steps"])


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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tasks(dataset_name: str) -> Dict[str, Dict[str, Any]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset_name == "HumanEval+":
        return get_human_eval_plus()
    if dataset_name == "MBPP+":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset_name}")


def _task_ids(split_plan: Dict[str, Any], dataset_name: str, assigned_split: str, max_tasks: int | None) -> List[str]:
    task_ids = sorted(
        str(row["task_id"])
        for row in split_plan["records"]
        if row.get("dataset") == dataset_name and row.get("assigned_split") == assigned_split
    )
    if max_tasks is not None:
        return task_ids[: int(max_tasks)]
    return task_ids


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


def _run_dir(output_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def _load_model(plan: Dict[str, Any], output_dir: Path, arm: str, seed: int | None, allow_download: bool) -> Any:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_source = plan["target_model"].get("snapshot_path") or plan["target_model"]["model_id"]
    model_kwargs: Dict[str, Any] = {}
    if not plan["target_model"].get("snapshot_path"):
        model_kwargs["revision"] = plan["target_model"].get("revision", "main")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=not allow_download,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        **model_kwargs,
    )
    if arm != "base_no_update":
        steps = _optimizer_steps(plan, arm)
        if seed is None:
            raise ValueError("seed is required for adapter sample generation")
        adapter_path = _run_dir(output_dir, arm, int(seed), steps)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model


def _load_tokenizer(plan: Dict[str, Any], allow_download: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer_source = plan["target_model"].get("snapshot_path") or plan["target_model"]["tokenizer_id"]
    tokenizer_kwargs: Dict[str, Any] = {}
    if not plan["target_model"].get("snapshot_path"):
        tokenizer_kwargs["revision"] = plan["target_model"].get("revision", "main")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        local_files_only=not allow_download,
        use_fast=True,
        **tokenizer_kwargs,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _sample_name(dataset_slug: str, arm: str, seed: int | None) -> str:
    suffix = "base" if seed is None else f"seed{seed}"
    return f"{dataset_slug}_{arm}_{suffix}.jsonl"


def _trim_completion(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.lstrip().startswith("python"):
            text = text.lstrip()[len("python") :]
    markers = ["\nif __name__ ==", "\n# Task", "\n###", "\n\n\n"]
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos > 0:
            cut = min(cut, pos)
    return text[:cut].rstrip() + "\n"


@torch.no_grad()
def generate_one(
    plan_path: Path,
    split_path: Path,
    output_dir: Path,
    arm: str,
    seed: int | None,
    datasets: List[str],
    max_tasks: int | None,
    max_new_tokens: int,
    generation_batch_size: int,
    allow_download: bool,
    overwrite: bool,
    progress_every: int,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    if arm != "base_no_update" and arm not in _trained_arms(plan):
        raise ValueError(f"Unknown arm for plan: {arm}")
    split_plan = load_json(split_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic EvalPlus generation.")
    _set_seed(0 if seed is None else int(seed))
    sample_dir = output_dir / "evalplus_guardrail" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "evalplus_guardrail" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(plan, allow_download)
    model = _load_model(plan, output_dir, arm, seed, allow_download)
    assigned_split = _stage_label(plan)
    started = time.time()
    outputs = []
    for dataset_name in datasets:
        dataset_slug, _ = DATASET_MAP[dataset_name]
        tasks = _load_tasks(dataset_name)
        task_ids = _task_ids(split_plan, dataset_name, assigned_split, max_tasks)
        path = sample_dir / _sample_name(dataset_slug, arm, seed)
        if path.exists() and not overwrite:
            existing = sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
            if existing == len(task_ids):
                outputs.append(
                    {
                        "dataset": dataset_name,
                        "dataset_slug": dataset_slug,
                        "path": str(path),
                        "task_count": len(task_ids),
                        "status": "already_complete",
                        "sha256": sha256_file(path),
                    }
                )
                continue
        with path.open("w", encoding="utf-8") as handle:
            for batch_start in range(0, len(task_ids), int(generation_batch_size)):
                batch_task_ids = task_ids[batch_start : batch_start + int(generation_batch_size)]
                prompts = [str(tasks[task_id]["prompt"]) for task_id in batch_task_ids]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(0)
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_width = encoded["input_ids"].shape[1]
                for task_id, sequence in zip(batch_task_ids, generated):
                    new_tokens = sequence[prompt_width:]
                    completion = _trim_completion(
                        tokenizer.decode(new_tokens, skip_special_tokens=True)
                    )
                    handle.write(
                        json.dumps(
                            {"task_id": task_id, "completion": completion},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                completed = min(batch_start + len(batch_task_ids), len(task_ids))
                if progress_every > 0 and (
                    batch_start == 0
                    or completed == len(task_ids)
                    or completed // progress_every != batch_start // progress_every
                ):
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "seed": seed,
                                "dataset": dataset_name,
                                "task": completed,
                                "total": len(task_ids),
                            }
                        ),
                        flush=True,
                    )
        outputs.append(
            {
                "dataset": dataset_name,
                "dataset_slug": dataset_slug,
                "path": str(path),
                "task_count": len(task_ids),
                "status": "generated",
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "code-domain-evalplus-samples-manifest-v1",
        "status": "evalplus_samples_generated",
        "arm": arm,
        "seed": seed,
        "plan_sha256": sha256_file(plan_path),
        "split_sha256": sha256_file(split_path),
        "datasets": outputs,
        "max_tasks_per_suite": max_tasks,
        "max_new_tokens": max_new_tokens,
        "generation_batch_size": generation_batch_size,
        "temperature": 0.0,
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "confirmatory_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": f"EvalPlus {assigned_split} sample generation only; Stage C guardrail evidence, never selector objective.",
    }
    manifest_name = f"{arm}_{'base' if seed is None else f'seed{seed}'}_manifest.json"
    save_json(manifest_dir / manifest_name, manifest)
    print(json.dumps({"status": manifest["status"], "arm": arm, "seed": seed, "datasets": outputs}, indent=2))
    return manifest


def generate_missing(
    plan_path: Path,
    split_path: Path,
    output_dir: Path,
    arms: List[str],
    seeds: List[int],
    datasets: List[str],
    max_tasks: int | None,
    max_new_tokens: int,
    generation_batch_size: int,
    max_runs: int | None,
    allow_download: bool,
    progress_every: int,
) -> Dict[str, Any]:
    tasks: List[Dict[str, Any]] = [{"arm": "base_no_update", "seed": None}]
    tasks.extend({"arm": arm, "seed": seed} for arm in arms for seed in seeds)
    executed = []
    skipped = []
    for task in tasks:
        if max_runs is not None and len(executed) >= max_runs:
            continue
        manifest = generate_one(
            plan_path,
            split_path,
            output_dir,
            str(task["arm"]),
            task["seed"],
            datasets,
            max_tasks,
            max_new_tokens,
            generation_batch_size,
            allow_download,
            overwrite=False,
            progress_every=progress_every,
        )
        if all(row["status"] == "already_complete" for row in manifest["datasets"]):
            skipped.append({"arm": task["arm"], "seed": task["seed"], "status": "already_complete"})
        else:
            executed.append({"arm": task["arm"], "seed": task["seed"], "status": manifest["status"]})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "schema_version": "code-domain-evalplus-generate-missing-summary-v1",
        "executed": executed,
        "skipped": skipped,
        "remaining_note": "Use the same command again; generation skips complete sample files.",
    }
    save_json(output_dir / "evalplus_guardrail" / "generate_missing_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate EvalPlus guardrail samples.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    common.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--datasets", default="HumanEval+,MBPP+")
    common.add_argument("--max-tasks", type=int)
    common.add_argument("--max-new-tokens", type=int, default=384)
    common.add_argument("--generation-batch-size", type=int, default=8)
    common.add_argument("--allow-download", action="store_true")
    common.add_argument("--progress-every", type=int, default=25)

    one = sub.add_parser("one", parents=[common])
    one.add_argument("--arm", required=True)
    one.add_argument("--seed", type=int)
    one.add_argument("--overwrite", action="store_true")

    missing = sub.add_parser("missing", parents=[common])
    missing.add_argument("--arms")
    missing.add_argument("--seeds")
    missing.add_argument("--max-runs", type=int)

    args = parser.parse_args()
    plan = load_json(args.plan)
    datasets = _parse_csv(args.datasets, DATASET_MAP)
    seeds = _parse_int_csv(
        getattr(args, "seeds", None),
        _training_seeds(plan),
    )
    if args.command == "one":
        generate_one(
            args.plan,
            args.split,
            args.output_dir,
            args.arm,
            args.seed,
            datasets,
            args.max_tasks,
            args.max_new_tokens,
            args.generation_batch_size,
            args.allow_download,
            args.overwrite,
            args.progress_every,
        )
    elif args.command == "missing":
        generate_missing(
            args.plan,
            args.split,
            args.output_dir,
            _parse_csv(args.arms, _trained_arms(plan)),
            seeds,
            datasets,
            args.max_tasks,
            args.max_new_tokens,
            args.generation_batch_size,
            args.max_runs,
            args.allow_download,
            args.progress_every,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
