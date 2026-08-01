#!/usr/bin/env python3
"""Freeze and materialize the target-size canonical redundancy development rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from safetensors.torch import save_file

from data_eval_common import save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROXY_PLAN = ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
DEFAULT_CANONICAL_DECISION = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json"
)
DEFAULT_OUTPUT_PLAN = (
    ROOT / "configs" / "temporal_code_redundancy_target_size_development_qwen3_4b_v1.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "redundancy_target_size_qwen3_4b_v1"
DEFAULT_MANIFEST = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_target_size_qwen3_4b_blocks_manifest.json"
)
QWEN3_SNAPSHOT = Path(
    r"D:\UNLV-Research\hf_cache\hub\models--Qwen--Qwen3-4B-Base\snapshots\906bfd4b4dc7f14ee4320094d8b41684abff8539"
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _pack_exact(
    source_path: Path,
    tokenizer: Any,
    *,
    exact_tokens: int,
    sequence_length: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if exact_tokens % sequence_length:
        raise ValueError(f"Token budget must divide sequence length: {exact_tokens} / {sequence_length}")
    eos = tokenizer.eos_token_id
    token_ids: List[int] = []
    source_records_read = 0
    nonempty_records_read = 0
    complete_records_consumed = 0
    final_record_uid = None
    final_record_tokens_available = 0
    final_record_tokens_consumed = 0
    cut_inside_final_record = False
    repository_ids = set()

    for row in _jsonl(source_path):
        source_records_read += 1
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        repository_ids.add(str(row.get("repository_identity") or row.get("repo") or "unknown"))
        nonempty_records_read += 1
        record_ids = list(tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            record_ids.append(int(eos))
        remaining = exact_tokens - len(token_ids)
        if remaining <= 0:
            break
        take = min(remaining, len(record_ids))
        token_ids.extend(int(value) for value in record_ids[:take])
        final_record_uid = str(row.get("chunk_uid") or row.get("record_id") or source_records_read)
        final_record_tokens_available = len(record_ids)
        final_record_tokens_consumed = take
        cut_inside_final_record = take < len(record_ids)
        if take == len(record_ids):
            complete_records_consumed += 1
        if len(token_ids) == exact_tokens:
            break

    if len(token_ids) != exact_tokens:
        raise RuntimeError(
            f"Insufficient tokens in {source_path}: required={exact_tokens}, got={len(token_ids)}"
        )
    tensor = torch.tensor(token_ids, dtype=torch.int32).reshape(
        exact_tokens // sequence_length,
        sequence_length,
    )
    return tensor.contiguous(), {
        "source_records_read": source_records_read,
        "nonempty_records_read": nonempty_records_read,
        "complete_records_consumed": complete_records_consumed,
        "final_record_uid": final_record_uid,
        "final_record_tokens_available": final_record_tokens_available,
        "final_record_tokens_consumed": final_record_tokens_consumed,
        "cut_inside_final_record": cut_inside_final_record,
        "repository_count_seen_before_budget": len(repository_ids),
        "exact_tokens": exact_tokens,
        "sequence_length": sequence_length,
        "blocks": exact_tokens // sequence_length,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "tensor_content_sha256": _tensor_sha256(tensor),
        "first_block_content_sha256": _tensor_sha256(tensor[0]),
        "last_block_content_sha256": _tensor_sha256(tensor[-1]),
        "minimum_token_id": int(tensor.min().item()),
        "maximum_token_id": int(tensor.max().item()),
    }


def _snapshot_artifacts(snapshot_path: Path) -> Dict[str, Dict[str, str]]:
    required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    return {
        name: {"path": str(snapshot_path / name), "sha256": sha256_file(snapshot_path / name)}
        for name in required
    }


def build(
    proxy_plan_path: Path,
    canonical_decision_path: Path,
    output_plan_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    proxy_plan = _load_json(proxy_plan_path)
    canonical_decision = _load_json(canonical_decision_path)
    if canonical_decision.get("status") != "canonical_qwen25_0p5b_development_guardrails_passed":
        raise RuntimeError("Canonical 0.5B development guardrails are not passed.")
    if not QWEN3_SNAPSHOT.exists():
        raise FileNotFoundError(f"Missing Qwen3-4B local snapshot: {QWEN3_SNAPSHOT}")

    sequence_length = 2048
    exact_train_tokens = 327680
    exact_eval_tokens = 65536
    seeds = [11, 23, 37]
    arms = {
        "binary_current_equal_budget": proxy_plan["arms"]["binary_current_equal_budget"],
        "stageA_random_common_disjoint_equal_budget": proxy_plan["arms"][
            "stageA_random_common_disjoint_equal_budget"
        ],
    }

    plan = {
        "schema_version": "temporal-code-redundancy-target-size-development-v1",
        "status": "frozen_before_target_size_development_outcomes",
        "purpose": (
            "Rerun the frozen canonical binary redundancy selector path on the target-size "
            "Qwen3-4B base model, without changing selector features or using Utility in Stage B."
        ),
        "source_sha256": {
            str(proxy_plan_path): sha256_file(proxy_plan_path),
            str(canonical_decision_path): sha256_file(canonical_decision_path),
        },
        "target_model": {
            "model_id": "Qwen/Qwen3-4B-Base",
            "tokenizer_id": "Qwen/Qwen3-4B-Base",
            "revision": "main",
            "snapshot_path": str(QWEN3_SNAPSHOT),
            "snapshot_artifacts": _snapshot_artifacts(QWEN3_SNAPSHOT),
        },
        "training_arms": list(arms.keys()),
        "primary_comparison": {
            "treatment": "binary_current_equal_budget",
            "primary_baseline": "stageA_random_common_disjoint_equal_budget",
            "base_reference": "base_no_update",
            "utility_scope": "Stage C validation only; Utility is forbidden in selector objectives.",
        },
        "arms": {
            name: {
                "path": contract["path"],
                "sha256": contract["sha256"],
                "role": (
                    "canonical_binary_selector"
                    if name == "binary_current_equal_budget"
                    else "common_selector_union_disjoint_stageA_random_baseline"
                ),
            }
            for name, contract in arms.items()
        },
        "training_recipe": {
            "method": "QLoRA continued pretraining",
            "quantization": "4-bit NF4 with double quantization",
            "compute_dtype": "bf16",
            "sequence_length": sequence_length,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "optimizer_steps": 40,
            "learning_rate": 0.00005,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "adapter": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "target_modules": "all-linear",
            },
            "development_training_seeds": seeds,
            "same_seed_set_for_every_arm": True,
            "exact_train_tokens_per_arm": exact_train_tokens,
            "shuffle_training_blocks_per_seed": True,
        },
        "heldout_nll": {
            "source_role": "same frozen development code heldout used by the 0.5B redundancy proxy",
            "path": proxy_plan["heldout_nll"]["path"],
            "sha256": proxy_plan["heldout_nll"]["sha256"],
            "exact_evaluation_tokens": exact_eval_tokens,
            "confirmatory_read_forbidden": True,
        },
        "required_stage_c_evidence": {
            "target_code_nll": "base plus all treatment/baseline seeds on frozen heldout blocks",
            "general_text_retention_nll": "required before promotion beyond development",
            "general_task_retention": "required before promotion beyond development",
            "evalplus_development": "required before promotion beyond development",
        },
        "decision_rule": {
            "primary_development_metric": "development code heldout mean NLL",
            "binary_vs_stageA_random_required_absolute_nll_reduction": 0.005,
            "seed_aggregation": "paired by seeds 11, 23, 37; report every seed and arithmetic mean",
            "guardrail_failure_action": "reject_target_size_promotion",
            "missing_required_evidence_action": "abstain",
        },
        "forbidden_uses": [
            "changing Stage-B selector objective using Utility or benchmark outcomes",
            "reintroducing log_count after the 0.5B futility decision",
            "using different random baselines per sensitivity arm",
            "reading untouched confirmatory outcomes before this development decision is frozen",
        ],
        "utility_scope": "Stage C validation only; never selector objective.",
        "confirmatory_outcomes_read": False,
        "claim_boundary": (
            "Target-size development freeze and block materialization only. This is not a "
            "production release, confirmatory result, or paper-level claim."
        ),
    }
    save_json(output_plan_path, plan)

    tokenizer = AutoTokenizer.from_pretrained(QWEN3_SNAPSHOT, local_files_only=True, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    output_blocks = output_dir / "token_blocks"
    output_blocks.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Dict[str, Any]] = {}
    for arm, contract in plan["arms"].items():
        source_path = Path(contract["path"])
        if sha256_file(source_path) != contract["sha256"]:
            raise RuntimeError(f"Frozen source hash mismatch for {arm}: {source_path}")
        tensor, audit = _pack_exact(
            source_path,
            tokenizer,
            exact_tokens=exact_train_tokens,
            sequence_length=sequence_length,
        )
        output = output_blocks / f"{arm}.safetensors"
        save_file({"input_ids": tensor}, output)
        artifacts[arm] = {
            "role": "training_arm",
            "path": str(output),
            "file_sha256": sha256_file(output),
            "source_path": str(source_path),
            "source_sha256": contract["sha256"],
            **audit,
        }

    heldout_source = Path(plan["heldout_nll"]["path"])
    if sha256_file(heldout_source) != plan["heldout_nll"]["sha256"]:
        raise RuntimeError(f"Frozen heldout hash mismatch: {heldout_source}")
    heldout_tensor, heldout_audit = _pack_exact(
        heldout_source,
        tokenizer,
        exact_tokens=exact_eval_tokens,
        sequence_length=sequence_length,
    )
    heldout_output = output_blocks / "development_code_nll_heldout.safetensors"
    save_file({"input_ids": heldout_tensor}, heldout_output)
    artifacts["development_code_nll_heldout"] = {
        "role": "heldout_nll",
        "path": str(heldout_output),
        "file_sha256": sha256_file(heldout_output),
        "source_path": str(heldout_source),
        "source_sha256": plan["heldout_nll"]["sha256"],
        **heldout_audit,
    }

    training_hashes = {
        value["tensor_content_sha256"]
        for value in artifacts.values()
        if value["role"] == "training_arm"
    }
    blockers: List[str] = []
    if len(training_hashes) != len(plan["training_arms"]):
        blockers.append("training_arm_token_tensors_are_not_unique")
    if artifacts["development_code_nll_heldout"]["tensor_content_sha256"] in training_hashes:
        blockers.append("heldout_tensor_matches_training_tensor")

    manifest = {
        "schema_version": "redundancy-target-size-qwen3-4b-blocks-manifest-v1",
        "status": "target_size_blocks_materialized" if not blockers else "target_size_blocks_blocked",
        "plan": {"path": str(output_plan_path), "sha256": sha256_file(output_plan_path)},
        "tokenizer": {
            "path": str(QWEN3_SNAPSHOT),
            "tokenizer_json_sha256": sha256_file(QWEN3_SNAPSHOT / "tokenizer.json"),
            "base_vocab_size": int(tokenizer.vocab_size),
            "tokenizer_size_with_added_tokens": int(len(tokenizer)),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "artifacts": artifacts,
        "training_contract": {
            "arm_count": len(plan["training_arms"]),
            "exact_tokens_per_arm": exact_train_tokens,
            "exact_blocks_per_arm": exact_train_tokens // sequence_length,
            "sequence_length": sequence_length,
            "seed_set": seeds,
            "optimizer_steps": plan["training_recipe"]["optimizer_steps"],
        },
        "heldout_contract": {
            "exact_tokens": exact_eval_tokens,
            "exact_blocks": exact_eval_tokens // sequence_length,
            "sequence_length": sequence_length,
        },
        "blockers": blockers,
        "utility_scope": plan["utility_scope"],
        "confirmatory_outcomes_read": False,
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze target-size Qwen3-4B redundancy development rerun.")
    parser.add_argument("--proxy-plan", type=Path, default=DEFAULT_PROXY_PLAN)
    parser.add_argument("--canonical-decision", type=Path, default=DEFAULT_CANONICAL_DECISION)
    parser.add_argument("--output-plan", type=Path, default=DEFAULT_OUTPUT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = build(
        args.proxy_plan,
        args.canonical_decision,
        args.output_plan,
        args.output_dir,
        args.manifest,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "plan": manifest["plan"],
                "artifacts": {
                    name: {
                        "role": row["role"],
                        "blocks": row["blocks"],
                        "exact_tokens": row["exact_tokens"],
                        "file_sha256": row["file_sha256"],
                    }
                    for name, row in manifest["artifacts"].items()
                },
                "blockers": manifest["blockers"],
            },
            indent=2,
        )
    )
    return 0 if not manifest["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
