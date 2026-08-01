#!/usr/bin/env python3
"""Evaluate frozen target and general-text NLL for redundancy proxy adapters."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_BLOCKS = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_packed_blocks_manifest.json"
)
DEFAULT_EVAL_INPUTS = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "redundancy_saturation_proxy_qwen25_0p5b_v1"
)


class BlockDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.blocks = load_file(path)["input_ids"].to(torch.long)

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        value = self.blocks[index]
        return {"input_ids": value, "labels": value.clone()}


def _device_summary() -> Dict[str, Any]:
    return {
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _load_model(plan: Dict[str, Any], adapter_path: Path | None) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        plan["target_model"]["snapshot_path"],
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            local_files_only=True,
        )
    model.eval()
    return model


@torch.no_grad()
def _mean_nll(model: Any, blocks_path: Path, batch_size: int) -> Dict[str, Any]:
    loader = DataLoader(
        BlockDataset(blocks_path),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    total_loss = 0.0
    predicted_tokens = 0
    blocks = 0
    for batch in loader:
        batch = {key: value.to(0) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        batch_predicted = int(
            batch["input_ids"].shape[0] * (batch["input_ids"].shape[1] - 1)
        )
        total_loss += float(loss.detach().cpu()) * batch_predicted
        predicted_tokens += batch_predicted
        blocks += int(batch["input_ids"].shape[0])
    mean_nll = total_loss / predicted_tokens
    return {
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else None,
        "blocks": blocks,
        "predicted_tokens": predicted_tokens,
        "block_file_sha256": sha256_file(blocks_path),
    }


def _result_path(
    output_dir: Path,
    arm: str,
    seed: int | None,
) -> Path:
    name = "base_no_update" if seed is None else f"{arm}_seed{seed}"
    return output_dir / "nll_evaluation" / f"{name}.json"


def _adapter_path(
    output_dir: Path,
    arm: str,
    seed: int,
    steps: int,
) -> Path:
    return output_dir / "qlora_runs" / f"{arm}_seed{seed}_steps{steps}"


def _completed(
    path: Path,
    target_sha: str,
    general_sha: str,
    plan_sha: str,
) -> bool:
    if not path.exists():
        return False
    try:
        row = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("status") == "redundancy_proxy_nll_evaluation_completed"
        and row.get("plan_sha256") == plan_sha
        and row.get("target_nll", {}).get("block_file_sha256") == target_sha
        and row.get("general_text_nll", {}).get("block_file_sha256") == general_sha
    )


def evaluate_one(
    plan_path: Path,
    blocks_manifest_path: Path,
    eval_inputs_path: Path,
    output_dir: Path,
    arm: str,
    seed: int | None,
    batch_size: int,
) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for proxy NLL evaluation.")
    torch.cuda.set_device(0)
    torch.cuda.init()
    plan = load_json(plan_path)
    blocks = load_json(blocks_manifest_path)
    eval_inputs = load_json(eval_inputs_path)
    steps = int(plan["training_recipe"]["optimizer_steps"])
    adapter_path = None
    adapter_manifest_sha = None
    if arm != "base_no_update":
        if seed is None:
            raise ValueError("seed is required for trained adapter evaluation")
        adapter_path = _adapter_path(output_dir, arm, seed, steps)
        run_result = load_json(adapter_path / "run_result.json")
        if run_result["status"] != "redundancy_proxy_qlora_completed":
            raise RuntimeError(f"Incomplete adapter run: {adapter_path}")
        adapter_manifest_sha = sha256_file(adapter_path / "adapter_manifest.json")

    target = blocks["artifacts"]["development_code_nll_heldout"]
    target_path = Path(target["path"])
    general = eval_inputs["general_text_retention"]["blocks"]
    general_path = Path(general["path"])
    if sha256_file(target_path) != target["file_sha256"]:
        raise RuntimeError("Target heldout block hash mismatch.")
    if sha256_file(general_path) != general["file_sha256"]:
        raise RuntimeError("General-text block hash mismatch.")

    started = time.time()
    model = _load_model(plan, adapter_path)
    target_result = _mean_nll(model, target_path, batch_size)
    general_result = _mean_nll(model, general_path, batch_size)
    result = {
        "schema_version": "redundancy-proxy-nll-evaluation-v1",
        "status": "redundancy_proxy_nll_evaluation_completed",
        "arm": arm,
        "seed": seed,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_manifest_sha256": adapter_manifest_sha,
        "target_nll": target_result,
        "general_text_nll": general_result,
        "batch_size": batch_size,
        "plan_sha256": sha256_file(plan_path),
        "block_manifest_sha256": sha256_file(blocks_manifest_path),
        "evaluation_inputs_sha256": sha256_file(eval_inputs_path),
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Frozen target and general-text NLL evidence only. General-task and "
            "EvalPlus guardrails remain required before a proxy decision."
        ),
    }
    output = _result_path(output_dir, arm, seed)
    save_json(output, result)
    print(json.dumps(result, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_missing(
    plan_path: Path,
    blocks_manifest_path: Path,
    eval_inputs_path: Path,
    output_dir: Path,
    batch_size: int,
    max_evals: int | None,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    blocks = load_json(blocks_manifest_path)
    eval_inputs = load_json(eval_inputs_path)
    target_sha = blocks["artifacts"]["development_code_nll_heldout"]["file_sha256"]
    general_sha = eval_inputs["general_text_retention"]["blocks"]["file_sha256"]
    plan_sha = sha256_file(plan_path)
    jobs = [("base_no_update", None)]
    jobs.extend(
        (arm, int(seed))
        for arm in plan["arms"]
        for seed in plan["training_recipe"]["seeds"]
    )
    executed = []
    skipped = []
    for arm, seed in jobs:
        output = _result_path(output_dir, arm, seed)
        if _completed(output, target_sha, general_sha, plan_sha):
            skipped.append({"arm": arm, "seed": seed})
            continue
        if max_evals is not None and len(executed) >= max_evals:
            continue
        result = evaluate_one(
            plan_path,
            blocks_manifest_path,
            eval_inputs_path,
            output_dir,
            arm,
            seed,
            batch_size,
        )
        executed.append({"arm": arm, "seed": seed, "status": result["status"]})
    remaining = [
        {"arm": arm, "seed": seed}
        for arm, seed in jobs
        if not _completed(
            _result_path(output_dir, arm, seed),
            target_sha,
            general_sha,
            plan_sha,
        )
    ]
    summary = {
        "schema_version": "redundancy-proxy-nll-evaluate-missing-summary-v1",
        "status": (
            "redundancy_proxy_nll_evaluation_complete"
            if not remaining
            else "redundancy_proxy_nll_evaluation_incomplete"
        ),
        "executed": executed,
        "skipped": skipped,
        "remaining": remaining,
    }
    save_json(output_dir / "nll_evaluation" / "evaluate_missing_summary.json", summary)
    return summary


def _paired_summary(deltas: List[float], t_critical: float) -> Dict[str, Any]:
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
    half_width = t_critical * sd / math.sqrt(len(deltas))
    return {
        "seed_deltas": deltas,
        "mean": mean,
        "sample_sd": sd,
        "one_sided_95_lower": mean - half_width,
        "one_sided_95_upper": mean + half_width,
        "paired_mde_95": half_width,
        "positive_seed_count": sum(delta > 0 for delta in deltas),
        "nonpositive_seed_count": sum(delta <= 0 for delta in deltas),
    }


def summarize(
    plan_path: Path,
    eval_inputs_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    plan = load_json(plan_path)
    eval_inputs = load_json(eval_inputs_path)
    results = {}
    base = load_json(_result_path(output_dir, "base_no_update", None))
    results["base_no_update"] = base
    for arm in plan["arms"]:
        results[arm] = {
            str(seed): load_json(_result_path(output_dir, arm, int(seed)))
            for seed in plan["training_recipe"]["seeds"]
        }
    seeds = [int(seed) for seed in plan["training_recipe"]["seeds"]]
    binary = "binary_current_equal_budget"
    candidate = "log_count_equal_budget"
    random_arm = "stageA_random_common_disjoint_equal_budget"
    curation_deltas = [
        results[random_arm][str(seed)]["target_nll"]["mean_nll"]
        - results[candidate][str(seed)]["target_nll"]["mean_nll"]
        for seed in seeds
    ]
    noninferiority_deltas = [
        results[candidate][str(seed)]["target_nll"]["mean_nll"]
        - results[binary][str(seed)]["target_nll"]["mean_nll"]
        for seed in seeds
    ]
    general_text_increases = {
        arm: [
            results[arm][str(seed)]["general_text_nll"]["mean_nll"]
            - base["general_text_nll"]["mean_nll"]
            for seed in seeds
        ]
        for arm in plan["arms"]
    }
    t_critical = 2.919986
    curation = _paired_summary(curation_deltas, t_critical)
    noninferiority = _paired_summary(noninferiority_deltas, t_critical)
    general_text = {
        arm: _paired_summary(deltas, t_critical)
        for arm, deltas in general_text_increases.items()
    }
    floor = float(plan["decision_contract"]["practical_absolute_nll_floor"])
    curation_pass = (
        curation["one_sided_95_lower"] >= floor
        and curation["positive_seed_count"] >= 2
    )
    noninferiority_pass = (
        noninferiority["one_sided_95_upper"] <= floor
        and noninferiority["nonpositive_seed_count"] >= 2
    )
    general_margin = float(
        eval_inputs["general_text_retention"]["maximum_allowed_mean_nll_increase"]
    )
    general_text_pass = all(
        row["one_sided_95_upper"] <= general_margin for row in general_text.values()
    )
    report = {
        "schema_version": "redundancy-proxy-nll-summary-v1",
        "status": "redundancy_proxy_nll_summary_ready",
        "target_nll_means": {
            arm: (
                base["target_nll"]["mean_nll"]
                if arm == "base_no_update"
                else statistics.mean(
                    results[arm][str(seed)]["target_nll"]["mean_nll"]
                    for seed in seeds
                )
            )
            for arm in ["base_no_update", *plan["arms"].keys()]
        },
        "general_text_nll_means": {
            arm: (
                base["general_text_nll"]["mean_nll"]
                if arm == "base_no_update"
                else statistics.mean(
                    results[arm][str(seed)]["general_text_nll"]["mean_nll"]
                    for seed in seeds
                )
            )
            for arm in ["base_no_update", *plan["arms"].keys()]
        },
        "curation_effect_random_minus_log_count": {
            **curation,
            "required_lower_bound": floor,
            "passed": curation_pass,
        },
        "candidate_noninferiority_log_count_minus_binary": {
            **noninferiority,
            "maximum_upper_bound": floor,
            "passed": noninferiority_pass,
        },
        "general_text_retention": {
            "margin": general_margin,
            "arms": general_text,
            "passed": general_text_pass,
        },
        "nll_gate_status": (
            "passed"
            if curation_pass and noninferiority_pass and general_text_pass
            else "failed"
        ),
        "final_proxy_decision": "not_available_missing_general_task_and_evalplus",
        "missing_required_guardrails": ["general_task_retention", "evalplus_code_retention"],
        "utility_scope": plan["primary_comparison"]["utility_scope"],
        "claim_boundary": (
            "Target and general-text NLL analysis only. Final proxy promotion "
            "remains unavailable until all frozen guardrails complete."
        ),
    }
    save_json(output_dir / "nll_evaluation" / "nll_summary.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate redundancy proxy NLL.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    parser.add_argument("--eval-inputs", type=Path, default=DEFAULT_EVAL_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("evaluate-one")
    one.add_argument("--arm", required=True)
    one.add_argument("--seed", type=int)
    one.add_argument("--batch-size", type=int, default=1)

    missing = sub.add_parser("evaluate-missing")
    missing.add_argument("--batch-size", type=int, default=1)
    missing.add_argument("--max-evals", type=int)

    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "evaluate-one":
        evaluate_one(
            args.plan,
            args.blocks,
            args.eval_inputs,
            args.output_dir,
            args.arm,
            args.seed,
            args.batch_size,
        )
    elif args.command == "evaluate-missing":
        result = evaluate_missing(
            args.plan,
            args.blocks,
            args.eval_inputs,
            args.output_dir,
            args.batch_size,
            args.max_evals,
        )
        print(json.dumps(result, indent=2))
    else:
        summarize(args.plan, args.eval_inputs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
