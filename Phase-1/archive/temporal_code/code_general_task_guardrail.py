#!/usr/bin/env python3
"""Run lm-eval general-task retention guardrails for code-domain arms."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


DEFAULT_PLAN = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
TASKS = ("hellaswag", "arc_challenge", "piqa", "winogrande")
TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "curated_v2_equal_budget",
    "known_high_quality_equal_budget",
)


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


def _run_dir(output_dir: Path, arm: str, seed: int, steps: int) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def _result_path(output_dir: Path, arm: str, seed: int | None, limit: int | float | None) -> Path:
    suffix = "base" if seed is None else f"seed{seed}"
    limit_suffix = "full" if limit is None else f"limit{str(limit).replace('.', 'p')}"
    return output_dir / "general_task_guardrail" / "lm_eval" / f"{arm}_{suffix}_{limit_suffix}.json"


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


def _to_plain_results(results: Any) -> Dict[str, Any]:
    if isinstance(results, dict):
        return results
    if hasattr(results, "to_dict"):
        return results.to_dict()
    raise TypeError(f"Unsupported lm-eval results object: {type(results)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _completed_tasks(raw_results: Dict[str, Any], requested_tasks: Iterable[str]) -> List[str]:
    results = raw_results.get("results") if isinstance(raw_results, dict) else {}
    if not isinstance(results, dict):
        return []
    requested = [str(task) for task in requested_tasks]
    return [task for task in requested if task in results]


def _merge_lm_eval_results(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    if not base:
        return copy.deepcopy(update)
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _covers_requested_tasks(row: Dict[str, Any], requested_tasks: Iterable[str]) -> bool:
    raw_results = row.get("lm_eval_results") if isinstance(row.get("lm_eval_results"), dict) else {}
    return set(_completed_tasks(raw_results, requested_tasks)) == {str(task) for task in requested_tasks}


def _normalize_suite_status(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(row)
    raw_results = (
        normalized.get("lm_eval_results")
        if isinstance(normalized.get("lm_eval_results"), dict)
        else {}
    )
    completed = _completed_tasks(raw_results, TASKS)
    completed_set = set(completed)
    normalized["tasks"] = list(TASKS)
    normalized["tasks_completed"] = completed
    normalized["tasks_remaining"] = [task for task in TASKS if task not in completed_set]
    normalized["status"] = (
        "general_task_lm_eval_completed"
        if not normalized["tasks_remaining"]
        else "general_task_lm_eval_partial"
    )
    return normalized


def _build_result_report(
    plan: Dict[str, Any],
    arm: str,
    seed: int | None,
    tasks: List[str],
    limit: int | float | None,
    batch_size: str,
    cache_requests: bool,
    raw_results: Dict[str, Any],
    source_sha256: Dict[str, str],
    started: float,
) -> Dict[str, Any]:
    tasks_completed = _completed_tasks(raw_results, tasks)
    completed = set(tasks_completed) == set(tasks)
    return {
        "schema_version": "code-domain-general-task-lm-eval-result-v1",
        "status": "general_task_lm_eval_completed" if completed else "general_task_lm_eval_partial",
        "arm": arm,
        "seed": seed,
        "tasks": tasks,
        "tasks_completed": tasks_completed,
        "tasks_remaining": [task for task in tasks if task not in set(tasks_completed)],
        "limit": limit,
        "batch_size": batch_size,
        "cache_requests": cache_requests,
        "lm_eval_results": raw_results,
        "source_sha256": source_sha256,
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "confirmatory_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "General-task retention guardrail only; Stage C evidence, never selector objective.",
    }


def _quantized_hflm_class() -> Any:
    from lm_eval.models.huggingface import HFLM

    class QuantizedHFLM(HFLM):
        def __init__(self, forced_quantization_config: Any, *args: Any, **kwargs: Any) -> None:
            self._forced_quantization_config = forced_quantization_config
            super().__init__(*args, **kwargs)

        def _create_model(
            self,
            *args: Any,
            quantization_config: Any | None = None,
            **kwargs: Any,
        ) -> None:
            return super()._create_model(
                *args,
                quantization_config=self._forced_quantization_config,
                **kwargs,
            )

    return QuantizedHFLM


def run_one(
    plan_path: Path,
    output_dir: Path,
    arm: str,
    seed: int | None,
    tasks: List[str],
    limit: int | float | None,
    batch_size: str,
    cache_requests: bool,
    overwrite: bool,
    max_task_runs: int | None = None,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    if arm != "base_no_update" and arm not in _trained_arms(plan):
        raise ValueError(f"Unknown arm for plan: {arm}")
    if arm != "base_no_update" and seed is None:
        raise ValueError("seed is required for trained arms")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for code-domain general-task evaluation.")
    output_path = _result_path(output_dir, arm, seed, limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_raw_results: Dict[str, Any] = {}
    if output_path.exists() and not overwrite:
        row = load_json(output_path)
        if _covers_requested_tasks(row, tasks):
            normalized = _normalize_suite_status(row)
            if normalized != row:
                save_json(output_path, normalized)
            print({"status": "already_complete", "path": str(output_path)})
            return normalized
        if row.get("status") in {"general_task_lm_eval_completed", "general_task_lm_eval_partial"}:
            prior_raw_results = row.get("lm_eval_results") if isinstance(row.get("lm_eval_results"), dict) else {}

    from lm_eval.evaluator import simple_evaluate
    from transformers import BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_source = plan["target_model"].get("snapshot_path") or plan["target_model"]["model_id"]
    tokenizer_source = plan["target_model"].get("snapshot_path") or plan["target_model"]["tokenizer_id"]
    model_kwargs: Dict[str, Any] = {
        "pretrained": model_source,
        "tokenizer": tokenizer_source,
        "dtype": "bfloat16",
        "device": "cuda",
        "batch_size": batch_size,
    }
    if not plan["target_model"].get("snapshot_path"):
        model_kwargs["revision"] = plan["target_model"].get("revision", "main")
    source_sha256 = {str(plan_path): sha256_file(plan_path)}
    if arm != "base_no_update":
        steps = int(_training_recipe(plan)["optimizer_steps"])
        adapter_path = _run_dir(output_dir, arm, int(seed), steps)
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing adapter: {adapter_path}")
        model_kwargs["peft"] = str(adapter_path)
        source_sha256[str(adapter_path / "adapter_config.json")] = sha256_file(
            adapter_path / "adapter_config.json"
        )
        source_sha256[str(adapter_path / "adapter_model.safetensors")] = sha256_file(
            adapter_path / "adapter_model.safetensors"
        )

    started = time.time()
    raw_results = copy.deepcopy(prior_raw_results)
    tasks_to_run = [task for task in tasks if task not in set(_completed_tasks(raw_results, tasks))]
    if max_task_runs is not None:
        tasks_to_run = tasks_to_run[: max(0, int(max_task_runs))]
    lm = None
    try:
        lm = _quantized_hflm_class()(quantization, **model_kwargs)
        for task in tasks_to_run:
            task_results = _json_safe(
                _to_plain_results(
                    simple_evaluate(
                        model=lm,
                        tasks=[task],
                        num_fewshot=0,
                        batch_size=batch_size,
                        device="cuda",
                        cache_requests=cache_requests,
                        limit=limit,
                        bootstrap_iters=0,
                        log_samples=False,
                        random_seed=0,
                        numpy_random_seed=1234,
                        torch_random_seed=1234,
                        fewshot_random_seed=1234,
                    )
                )
            )
            raw_results = _merge_lm_eval_results(raw_results, task_results)
            interim = _build_result_report(
                plan,
                arm,
                seed,
                list(TASKS),
                limit,
                batch_size,
                cache_requests,
                raw_results,
                source_sha256,
                started,
            )
            save_json(output_path, interim)
            print(
                {
                    "status": interim["status"],
                    "arm": arm,
                    "seed": seed,
                    "task_completed": task,
                    "tasks_completed": interim["tasks_completed"],
                    "tasks_remaining": interim["tasks_remaining"],
                    "path": str(output_path),
                    "elapsed_seconds": interim["elapsed_seconds"],
                }
            )
    finally:
        if lm is not None:
            del lm
        gc.collect()
        torch.cuda.empty_cache()
    report = _build_result_report(
        plan,
        arm,
        seed,
        list(TASKS),
        limit,
        batch_size,
        cache_requests,
        raw_results,
        source_sha256,
        started,
    )
    save_json(output_path, report)
    print(
        {
            "status": report["status"],
            "arm": arm,
            "seed": seed,
            "path": str(output_path),
            "elapsed_seconds": report["elapsed_seconds"],
        }
    )
    return report


def run_missing(
    plan_path: Path,
    output_dir: Path,
    arms: List[str],
    seeds: List[int],
    tasks: List[str],
    limit: int | float | None,
    batch_size: str,
    cache_requests: bool,
    max_runs: int | None,
    max_task_runs: int | None = None,
) -> Dict[str, Any]:
    jobs: List[Dict[str, Any]] = [{"arm": "base_no_update", "seed": None}]
    jobs.extend({"arm": arm, "seed": seed} for arm in arms for seed in seeds)
    executed = []
    skipped = []
    for job in jobs:
        if max_runs is not None and len(executed) >= max_runs:
            continue
        path = _result_path(output_dir, str(job["arm"]), job["seed"], limit)
        if path.exists():
            row = load_json(path)
            if row.get("status") == "general_task_lm_eval_completed" and _covers_requested_tasks(row, tasks):
                skipped.append({"arm": job["arm"], "seed": job["seed"], "status": "already_complete"})
                continue
        row = run_one(
            plan_path,
            output_dir,
            str(job["arm"]),
            job["seed"],
            tasks,
            limit,
            batch_size,
            cache_requests,
            overwrite=False,
            max_task_runs=max_task_runs,
        )
        executed.append(
            {
                "arm": row["arm"],
                "seed": row["seed"],
                "status": row["status"],
                "tasks_completed": row.get("tasks_completed"),
                "tasks_remaining": row.get("tasks_remaining"),
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
    summary = {
        "schema_version": "code-domain-general-task-lm-eval-missing-summary-v1",
        "status": "general_task_lm_eval_missing_completed",
        "executed": executed,
        "skipped": skipped,
        "remaining_note": "Use the same command again; completed result files are skipped.",
    }
    save_json(output_dir / "general_task_guardrail" / "lm_eval_missing_summary.json", summary)
    print(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run code-domain general-task retention guardrails.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    common.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    common.add_argument("--tasks", default=",".join(TASKS))
    common.add_argument("--limit", type=float)
    common.add_argument("--batch-size", default="1")
    common.add_argument("--no-cache-requests", action="store_true")
    common.add_argument("--max-task-runs", type=int)

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
    tasks = _parse_csv(args.tasks, TASKS)
    limit: int | float | None
    if args.limit is None:
        limit = None
    elif float(args.limit).is_integer():
        limit = int(args.limit)
    else:
        limit = float(args.limit)
    if args.command == "one":
        run_one(
            args.plan,
            args.output_dir,
            args.arm,
            args.seed,
            tasks,
            limit,
            args.batch_size,
            not args.no_cache_requests,
            args.overwrite,
            args.max_task_runs,
        )
    elif args.command == "missing":
        seeds = _parse_int_csv(
            args.seeds,
            _training_seeds(plan),
        )
        run_missing(
            args.plan,
            args.output_dir,
            _parse_csv(args.arms, _trained_arms(plan)),
            seeds,
            tasks,
            limit,
            args.batch_size,
            not args.no_cache_requests,
            args.max_runs,
            args.max_task_runs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
