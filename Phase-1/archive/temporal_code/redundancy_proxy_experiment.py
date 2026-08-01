#!/usr/bin/env python3
"""Freeze the small-model redundancy-saturation proxy experiment contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARMS_DIR = (
    OUTPUT_DIR / "temporal_code_collection" / "redundancy_saturation_proxy_arms_v1"
)
DEFAULT_HELDOUT = (
    OUTPUT_DIR
    / "code_domain_v2_development_qwen3_4b"
    / "heldouts"
    / "development_code_nll_heldout.jsonl"
)
DEFAULT_MODEL_SNAPSHOT = Path(
    "D:/UNLV-Research/hf_cache/hub/models--Qwen--Qwen2.5-0.5B/"
    "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
)
DEFAULT_CONFIG = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
DEFAULT_REPORT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_experiment_freeze_report.json"
)

ARMS = {
    "binary_current_equal_budget": "binary_current_equal_budget.jsonl",
    "log_count_equal_budget": "log_count_equal_budget.jsonl",
    "stageA_random_common_disjoint_equal_budget": (
        "stageA_random_common_disjoint_equal_budget.jsonl"
    ),
}


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _repository(row: Dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    return str(
        provenance.get("repository_identity")
        or row.get("repository_identity")
        or ""
    )


def _token_count(path: Path, tokenizer: Any) -> int:
    total = 0
    eos = tokenizer.eos_token_id
    for row in _jsonl(path):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        total += len(tokenizer(text, add_special_tokens=False).input_ids)
        if eos is not None:
            total += 1
    return total


def _file_contract(path: Path) -> Dict[str, Any]:
    rows = list(_jsonl(path))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_count": len(rows),
        "repository_count": len({_repository(row) for row in rows if _repository(row)}),
        "repository_ids": sorted({_repository(row) for row in rows if _repository(row)}),
    }


def freeze(
    arms_dir: Path,
    heldout_path: Path,
    model_snapshot: Path,
    config_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_snapshot,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Frozen tokenizer has no EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sequence_length = 1024
    micro_batch_size = 1
    gradient_accumulation_steps = 8
    optimizer_steps = 40
    exact_train_tokens = (
        sequence_length
        * micro_batch_size
        * gradient_accumulation_steps
        * optimizer_steps
    )
    seeds = [11, 23, 37]

    arm_contracts: Dict[str, Dict[str, Any]] = {}
    blockers = []
    for arm, filename in ARMS.items():
        path = arms_dir / filename
        contract = _file_contract(path)
        raw_tokens = _token_count(path, tokenizer)
        contract.update(
            {
                "raw_tokenizer_tokens_with_eos": raw_tokens,
                "complete_sequence_blocks_available": raw_tokens // sequence_length,
                "exact_training_tokens": exact_train_tokens,
                "exact_training_blocks": exact_train_tokens // sequence_length,
                "tail_tokens_after_training_budget": raw_tokens - exact_train_tokens,
            }
        )
        if raw_tokens < exact_train_tokens:
            blockers.append(f"insufficient_tokenizer_tokens:{arm}")
        arm_contracts[arm] = contract

    heldout = _file_contract(heldout_path)
    heldout_raw_tokens = _token_count(heldout_path, tokenizer)
    heldout_blocks = heldout_raw_tokens // sequence_length
    exact_heldout_tokens = heldout_blocks * sequence_length
    train_repositories = set().union(
        *(set(contract["repository_ids"]) for contract in arm_contracts.values())
    )
    heldout_repositories = set(heldout["repository_ids"])
    overlap = sorted(train_repositories.intersection(heldout_repositories))
    heldout.update(
        {
            "source_split": "development",
            "allowed_content_types": ["code", "test"],
            "raw_tokenizer_tokens_with_eos": heldout_raw_tokens,
            "exact_evaluation_tokens": exact_heldout_tokens,
            "exact_evaluation_blocks": heldout_blocks,
            "dropped_tail_tokens": heldout_raw_tokens - exact_heldout_tokens,
            "train_repository_overlap_count": len(overlap),
            "train_repository_overlap": overlap,
        }
    )
    if overlap:
        blockers.append("heldout_repository_overlap")
    if exact_heldout_tokens <= 0:
        blockers.append("heldout_has_no_complete_blocks")

    snapshot_files = {}
    for filename in ("config.json", "tokenizer.json", "model.safetensors"):
        path = model_snapshot / filename
        if not path.exists():
            blockers.append(f"missing_model_artifact:{filename}")
            continue
        snapshot_files[filename] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    practical_floor = 0.002
    one_sided_t_95_df2 = 2.919986
    plan = {
        "schema_version": "temporal-code-redundancy-proxy-experiment-v1",
        "status": (
            "frozen_before_proxy_training_outcomes"
            if not blockers
            else "freeze_blocked"
        ),
        "purpose": (
            "Test whether count-sensitive Stage-B redundancy saturation can replace "
            "the current binary saturation evidence without using Utility in selection."
        ),
        "target_model": {
            "model_id": "Qwen/Qwen2.5-0.5B",
            "tokenizer_id": "Qwen/Qwen2.5-0.5B",
            "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "parameter_count": 490000000,
            "snapshot_path": str(model_snapshot),
            "snapshot_artifacts": snapshot_files,
            "local_files_only": True,
        },
        "arms": arm_contracts,
        "primary_comparison": {
            "candidate": "log_count_equal_budget",
            "canonical_control": "binary_current_equal_budget",
            "operational_baseline": "stageA_random_common_disjoint_equal_budget",
            "utility_scope": "Stage C validation only; never selector objective",
        },
        "tokenization_and_packing": {
            "record_order": "existing JSONL order",
            "add_special_tokens": False,
            "append_eos_after_each_nonempty_record": True,
            "pad_in_training_stream": False,
            "sequence_length": sequence_length,
            "exact_train_tokens_per_arm": exact_train_tokens,
            "exact_train_blocks_per_arm": exact_train_tokens // sequence_length,
            "overflow_rule": "stop exactly at the common token budget",
            "tail_rule": "discard tokens after the common budget; never pad a partial block",
        },
        "training_recipe": {
            "method": "QLoRA continued pretraining",
            "quantization": "4-bit NF4 with double quantization",
            "compute_dtype": "bf16",
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer_steps": optimizer_steps,
            "effective_sequences_per_step": (
                micro_batch_size * gradient_accumulation_steps
            ),
            "learning_rate": 0.00005,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": True,
            "adapter": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": "all-linear",
            },
            "seeds": seeds,
            "same_seed_set_for_every_arm": True,
            "shuffle_training_blocks_per_seed": True,
        },
        "heldout_nll": heldout,
        "decision_contract": {
            "primary_metric": "mean causal-LM NLL on the frozen heldout blocks",
            "direction": "lower_is_better",
            "paired_unit": "training seed",
            "seed_count": len(seeds),
            "practical_absolute_nll_floor": practical_floor,
            "paired_mde_95_formula": (
                "2.919986 * sample_sd(seed_level_paired_delta) / sqrt(3)"
            ),
            "paired_mde_interpretation": (
                "Effects below max(0.002, paired_mde_95) are inconclusive and "
                "cannot support a positive Utility claim."
            ),
            "curation_effect": {
                "estimand": (
                    "NLL(stageA_random_common_disjoint) - NLL(log_count)"
                ),
                "pass_rule": (
                    "one-sided paired 95% lower confidence bound must be >= 0.002 "
                    "and at least 2 of 3 seed deltas must be positive"
                ),
            },
            "candidate_noninferiority": {
                "estimand": "NLL(log_count) - NLL(binary_current)",
                "margin": practical_floor,
                "pass_rule": (
                    "one-sided paired 95% upper confidence bound must be <= 0.002 "
                    "and at least 2 of 3 seeds must be non-worse"
                ),
            },
            "mechanism_requirement": (
                "The pre-frozen template-saturation diagnostic must improve or "
                "remain tied versus binary_current; missing diagnostic means abstain."
            ),
            "retention_requirement": (
                "All frozen general-text, general-task, and code-retention guardrails "
                "must pass; missing evidence means abstain."
            ),
            "promotion_rule": (
                "Promote log_count to Qwen3-4B development only if curation_effect, "
                "candidate_noninferiority, mechanism_requirement, and all retention "
                "guardrails pass. Otherwise hold, reject, or abstain without tuning "
                "this frozen proxy cycle."
            ),
        },
        "forbidden_uses": [
            "Utility or model outcomes in Stage-B scoring",
            "changing arms, seeds, token budget, heldout, or margins after outcomes",
            "reading Qwen3-4B outcomes to revise this proxy decision",
            "claiming superiority from a mean difference below the detectable-effect rule",
        ],
        "blockers": blockers,
        "claim_boundary": (
            "Pre-training proxy contract only. It freezes inputs and decisions but "
            "contains no Utility, retention, release, or framework-validity result."
        ),
    }
    save_json(config_path, plan)

    report = {
        "schema_version": "redundancy-proxy-experiment-freeze-report-v1",
        "status": (
            "redundancy_proxy_experiment_frozen"
            if not blockers
            else "redundancy_proxy_experiment_freeze_blocked"
        ),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "model_revision": plan["target_model"]["revision"],
        "seeds": seeds,
        "sequence_length": sequence_length,
        "exact_train_tokens_per_arm": exact_train_tokens,
        "exact_train_blocks_per_arm": exact_train_tokens // sequence_length,
        "exact_heldout_tokens": exact_heldout_tokens,
        "exact_heldout_blocks": heldout_blocks,
        "heldout_train_repository_overlap_count": len(overlap),
        "arm_raw_tokenizer_tokens": {
            arm: contract["raw_tokenizer_tokens_with_eos"]
            for arm, contract in arm_contracts.items()
        },
        "practical_absolute_nll_floor": practical_floor,
        "one_sided_t_critical_95_df2": one_sided_t_95_df2,
        "blockers": blockers,
        "claim_boundary": plan["claim_boundary"],
    }
    save_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the redundancy-saturation proxy experiment."
    )
    parser.add_argument("--arms-dir", type=Path, default=DEFAULT_ARMS_DIR)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_HELDOUT)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--config-output", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = freeze(
        args.arms_dir,
        args.heldout,
        args.model_snapshot,
        args.config_output,
        args.report_output,
    )
    print(report)
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
