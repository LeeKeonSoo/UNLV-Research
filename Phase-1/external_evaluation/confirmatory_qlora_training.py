#!/usr/bin/env python3
"""Run one frozen natural-budget QLoRA confirmatory training arm."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "code_7m_normal_hard_confirmatory_v1.json"
DEFAULT_INPUT_REPORT = Path(
    "D:/UNLV-Research/code_5m_corpus_v2/hard_confirmatory_7m_v1/training_inputs/training_inputs_report.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def resolve_run(
    protocol_path: Path, input_report_path: Path, arm: str, seed: int
) -> dict[str, Any]:
    """Validate the frozen external inputs for one arm before model allocation."""
    protocol = load_json(protocol_path)
    if protocol.get("status") != "frozen_before_curation_and_tokenizer_materialization":
        raise RuntimeError("Confirmatory protocol is not frozen before execution")
    training = protocol.get("training")
    if not isinstance(training, dict):
        raise TypeError("Confirmatory protocol has no training section")
    if arm not in training.get("arms", []):
        raise ValueError(f"Unknown confirmatory arm: {arm}")
    if seed not in training.get("seeds", []):
        raise ValueError(f"Seed is not preregistered: {seed}")
    report = load_json(input_report_path)
    if report.get("status") != "tokenizer_materialization_complete":
        raise RuntimeError("Tokenizer materialization is incomplete")
    arms = report.get("arms")
    if not isinstance(arms, dict) or not isinstance(arms.get(arm), dict):
        raise RuntimeError(f"Missing materialized training input for {arm}")
    arm_report = arms[arm]
    blocks_path = Path(str(arm_report["blocks_path"]))
    if not blocks_path.is_file() or sha256_file(blocks_path) != arm_report.get("blocks_sha256"):
        raise RuntimeError(f"Frozen token blocks are missing or changed for {arm}")
    output_root = Path(str(training["output_root"]))
    return {
        "protocol_path": protocol_path,
        "input_report_path": input_report_path,
        "protocol": protocol,
        "training": training,
        "input_report": report,
        "arm_report": arm_report,
        "blocks_path": blocks_path,
        "run_dir": output_root / "qlora_runs" / f"{arm}_seed{seed}_steps{arm_report['optimizer_steps']}",
        "seed": seed,
        "arm": arm,
    }


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def train(run: dict[str, Any], gpu: int) -> Path:
    """Train and save one complete adapter; incomplete runs never look complete."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA confirmatory training requires CUDA")
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"CUDA device {gpu} is unavailable")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    training = run["training"]
    arm_report = run["arm_report"]
    run_dir = Path(run["run_dir"])
    if (run_dir / "run_result.json").is_file():
        raise RuntimeError(f"Completed run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(f"cuda:{gpu}")
    set_seed(int(run["seed"]))
    blocks = torch.load(run["blocks_path"], map_location="cpu", weights_only=True)["input_ids"]
    if int(blocks.shape[0]) != int(arm_report["blocks"]):
        raise RuntimeError("Frozen block count does not match the materialization report")
    accumulation = int(training["gradient_accumulation_steps"])
    steps = int(arm_report["optimizer_steps"])
    if int(blocks.shape[0]) != steps * accumulation:
        raise RuntimeError("Frozen block grouping does not match optimizer steps")
    manifest = {
        "schema_version": "confirmatory-qlora-run-manifest-v1",
        "status": "running",
        "protocol_path": str(run["protocol_path"]),
        "protocol_sha256": sha256_file(Path(run["protocol_path"])),
        "training_input_report": str(run["input_report_path"]),
        "training_input_report_sha256": sha256_file(Path(run["input_report_path"])),
        "arm": run["arm"],
        "seed": run["seed"],
        "gpu": gpu,
        "blocks_path": str(run["blocks_path"]),
        "blocks_sha256": arm_report["blocks_sha256"],
        "materialized_tokens": arm_report["materialized_tokens"],
        "optimizer_steps": steps,
        "recipe": {
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "max_grad_norm": training["max_grad_norm"],
            "micro_batch_size": training["micro_batch_size"],
            "gradient_accumulation_steps": accumulation,
            "adapter": training["adapter"],
            "quantization": "4-bit NF4, double quantization, bfloat16 compute",
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(training["snapshot_path"]),
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": gpu},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    adapter = training["adapter"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(adapter["rank"]),
            lora_alpha=int(adapter["alpha"]),
            lora_dropout=float(adapter["dropout"]),
            target_modules=str(adapter["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    started = time.time()
    progress_path = run_dir / "progress.json"
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for offset in range(accumulation):
            batch = blocks[step * accumulation + offset].unsqueeze(0).to(device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            (loss / accumulation).backward()
            update_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            write_json(
                progress_path,
                {
                    "status": "running",
                    "completed_optimizer_steps": step + 1,
                    "total_optimizer_steps": steps,
                    "mean_microbatch_loss": update_loss / accumulation,
                    "elapsed_seconds": round(time.time() - started, 3),
                },
            )
    model.save_pretrained(run_dir)
    result = {
        **manifest,
        "status": "complete",
        "elapsed_seconds": round(time.time() - started, 3),
        "adapter_directory": str(run_dir),
    }
    write_json(run_dir / "run_result.json", result)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen QLoRA confirmatory training arm.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = resolve_run(args.protocol, args.input_report, args.arm, args.seed)
    if args.dry_run:
        print(json.dumps({key: str(value) for key, value in run.items() if key in {"arm", "seed", "blocks_path", "run_dir"}}, indent=2))
        return 0
    print(train(run, args.gpu))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
