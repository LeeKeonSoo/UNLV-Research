#!/usr/bin/env python3
"""Evaluate frozen Math transfer arms with the official lm-eval tasks."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Final, Protocol

# Importing Arrow before CUDA avoids a Windows DLL-loader conflict.
import pyarrow  # noqa: F401

from external_evaluation.evalplus_generator import (
    benchmark_root,
    load_json,
    load_model,
)


ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = ROOT / "protocols" / "math_7m_transfer_pilot_evaluation_v1.json"
DEFAULT_INPUT_REPORT: Final = Path(
    "D:/UNLV-Research/math_7m_framework_v1/"
    "training_inputs_benchmark_clean_v1/training_inputs_report.json"
)
TASKS: Final = ("gsm8k_cot_zeroshot", "hendrycks_math500")
ARMS: Final = (
    "base_no_update",
    "raw_math_natural",
    "normal_math_natural",
    "hard_math_natural",
    "nemo_math_natural",
)


class GenerationConfig(Protocol):
    max_new_tokens: int | None


class ModelWithGenerationConfig(Protocol):
    generation_config: GenerationConfig


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    arm: str
    seed: int | None
    limit: int | None
    task: str | None


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    protocol_path: Path
    input_report_path: Path
    identity: EvaluationIdentity
    batch_size: int
    bootstrap_iters: int


def resolve_tasks(task: str | None) -> tuple[str, ...]:
    if task is None:
        return TASKS
    if task not in TASKS:
        raise ValueError(f"Unsupported Math benchmark: {task}")
    return (task,)


def output_path(
    run_root: Path,
    identity: EvaluationIdentity,
) -> Path:
    seed_label = "base" if identity.seed is None else f"seed{identity.seed}"
    task_label = "" if identity.task is None else f"_{identity.task}"
    limit_label = "" if identity.limit is None else f"_limit{identity.limit}"
    return (
        run_root
        / "math_lm_eval"
        / f"{identity.arm}_{seed_label}{task_label}{limit_label}.json"
    )


def clear_pretrained_generation_limit(model: ModelWithGenerationConfig) -> None:
    """Let each official task supply its own maximum generation length."""
    model.generation_config.max_new_tokens = None


def validate_batch_size(value: str | int) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError("Evaluation batch size must be positive")
    return parsed


def evaluate(run: EvaluationRun) -> Path:
    """Run deterministic official Math tasks with the shared NF4 model loader."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import handle_non_serializable

    protocol = load_json(run.protocol_path)
    input_report = load_json(run.input_report_path)
    model, tokenizer = load_model(
        protocol,
        input_report,
        run.identity.arm,
        run.identity.seed,
    )
    clear_pretrained_generation_limit(model)
    harness_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=run.batch_size,
    )
    results = simple_evaluate(
        model=harness_model,
        tasks=list(resolve_tasks(run.identity.task)),
        batch_size=run.batch_size,
        limit=run.identity.limit,
        bootstrap_iters=run.bootstrap_iters,
        log_samples=True,
        random_seed=1234,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no Math evaluation result")
    results["unlv_evaluation"] = {
        "protocol": str(run.protocol_path),
        "arm": run.identity.arm,
        "seed": run.identity.seed,
        "tasks": list(resolve_tasks(run.identity.task)),
        "limit": run.identity.limit,
        "model_quantization": "4-bit NF4 double-quantization, bfloat16 compute",
        "batch_size": run.batch_size,
        "bootstrap_iters": run.bootstrap_iters,
    }
    target = output_path(benchmark_root(protocol), run.identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
            default=handle_non_serializable,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=validate_batch_size, default=4)
    parser.add_argument("--bootstrap-iters", type=int, default=1_000)
    args = parser.parse_args()
    if args.arm == "base_no_update" and args.seed is not None:
        parser.error("base_no_update does not accept --seed")
    if args.arm != "base_no_update" and args.seed is None:
        parser.error("adapter arms require --seed")
    print(
        evaluate(
            EvaluationRun(
                protocol_path=args.protocol,
                input_report_path=args.input_report,
                identity=EvaluationIdentity(args.arm, args.seed, args.limit, args.task),
                batch_size=args.batch_size,
                bootstrap_iters=args.bootstrap_iters,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
