#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Final

# Transformers may load Arrow after CUDA on Windows and crash in the DLL loader.
import pyarrow  # noqa: F401
import torch

from external_evaluation.runtime_paths import BenchmarkWorkerPaths


ROOT: Final = Path(__file__).resolve().parents[1]
PROTOCOL_PATH: Final = ROOT / "protocols" / "code_7m_normal_hard_confirmatory_v1.json"
INPUT_REPORT_PATH: Final = BenchmarkWorkerPaths.from_environment().input_report
DATASETS: Final = ("HumanEval+", "MBPP+")
ARMS: Final = (
    "base_no_update",
    "raw_audited_natural",
    "normal_natural",
    "hard_natural",
)


def resolve_datasets(dataset: str | None) -> tuple[str, ...]:
    if dataset is None:
        return DATASETS
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")
    return (dataset,)


def adapter_directory(run_root: Path, arm: str, seed: int, steps: int) -> Path:
    return run_root / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def benchmark_root(protocol: dict[str, Any]) -> Path:
    training = protocol.get("training")
    if not isinstance(training, dict):
        raise TypeError("Confirmatory protocol has no training section")
    return BenchmarkWorkerPaths.from_environment().benchmark_root(
        Path(str(training["output_root"]))
    )


def resolve_model_run(
    protocol: dict[str, Any],
    input_report: dict[str, Any],
    arm: str,
    seed: int | None,
) -> dict[str, Any]:
    training = protocol.get("training")
    if not isinstance(training, dict):
        raise TypeError("Confirmatory protocol has no training section")
    run_root = BenchmarkWorkerPaths.from_environment().training_output_root(
        Path(str(training["output_root"]))
    )
    if arm == "base_no_update":
        if seed is not None:
            raise ValueError("base_no_update does not accept a seed")
        return {"arm": arm, "seed": None, "adapter_path": None, "run_root": run_root}
    if arm not in training.get("arms", ()):
        raise ValueError(f"Unknown confirmatory arm: {arm}")
    if seed not in training.get("seeds", ()):
        raise ValueError(f"Seed is not preregistered: {seed}")
    reports = input_report.get("arms")
    if not isinstance(reports, dict) or not isinstance(reports.get(arm), dict):
        raise RuntimeError(f"Missing tokenizer materialization report for {arm}")
    steps = int(reports[arm]["optimizer_steps"])
    return {
        "arm": arm,
        "seed": seed,
        "adapter_path": adapter_directory(run_root, arm, int(seed), steps),
        "run_root": run_root,
    }


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return raw


def load_tasks(dataset: str) -> dict[str, dict[str, Any]]:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    if dataset == "HumanEval+":
        return get_human_eval_plus()
    if dataset == "MBPP+":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported EvalPlus dataset: {dataset}")


def load_model(
    protocol: dict[str, Any],
    input_report: dict[str, Any],
    arm: str,
    seed: int | None,
) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    training = protocol.get("training")
    if not isinstance(training, dict):
        raise TypeError("Invalid frozen confirmatory protocol")
    snapshot = str(
        BenchmarkWorkerPaths.from_environment().model_snapshot(
            Path(str(training["snapshot_path"]))
        )
    )
    resolved = resolve_model_run(protocol, input_report, arm, seed)
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
    adapter_path = resolved["adapter_path"]
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if not (adapter_path / "run_result.json").is_file():
            raise FileNotFoundError(f"Adapter is not complete: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.eval()
    return model, tokenizer


def _output_path(root: Path, dataset: str, arm: str, seed: int | None) -> Path:
    slug = "humaneval" if dataset == "HumanEval+" else "mbpp"
    suffix = "base" if seed is None else f"seed{seed}"
    return root / "evalplus" / f"{slug}_{arm}_{suffix}.jsonl"


@torch.no_grad()
def generate(
    arm: str,
    seed: int | None,
    batch_size: int,
    max_new_tokens: int,
    protocol_path: Path = PROTOCOL_PATH,
    input_report_path: Path = INPUT_REPORT_PATH,
    dataset: str | None = None,
) -> list[Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("EvalPlus generation requires CUDA")
    protocol = load_json(protocol_path)
    input_report = load_json(input_report_path)
    resolved = resolve_model_run(protocol, input_report, arm, seed)
    actual_seed = 0 if seed is None else seed
    random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    samples_root = benchmark_root(protocol) / "samples"
    requested = resolve_datasets(dataset)
    completed = [_output_path(samples_root, item, arm, seed) for item in requested]
    pending = [item for item, output in zip(requested, completed, strict=True) if not output.is_file()]
    if not pending:
        return completed
    model, tokenizer = load_model(protocol, input_report, arm, seed)
    for item in pending:
        tasks = load_tasks(item)
        task_ids = sorted(tasks)
        output = _output_path(samples_root, item, arm, seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for start in range(0, len(task_ids), batch_size):
                batch_ids = task_ids[start : start + batch_size]
                prompts = [str(tasks[task_id]["prompt"]) for task_id in batch_ids]
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(0)
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_width = encoded["input_ids"].shape[1]
                for task_id, sequence in zip(batch_ids, generated, strict=True):
                    completion = trim_completion(
                        tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True)
                    )
                    handle.write(json.dumps({"task_id": task_id, "completion": completion}) + "\n")
        os.replace(temporary, output)
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate frozen EvalPlus samples.")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--input-report", type=Path, default=INPUT_REPORT_PATH)
    args = parser.parse_args()
    if args.arm == "base_no_update" and args.seed is not None:
        parser.error("base_no_update does not accept --seed")
    if args.arm != "base_no_update" and args.seed is None:
        parser.error("adapter arms require --seed")
    for output in generate(
        arm=args.arm,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        protocol_path=args.protocol,
        input_report_path=args.input_report,
        dataset=args.dataset,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
