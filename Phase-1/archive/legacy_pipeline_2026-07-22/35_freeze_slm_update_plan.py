#!/usr/bin/env python3
"""Freeze tokenizer-budget and training contract for an SLM update experiment."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import iter_jsonl_records_resilient, load_json, save_json


DEFAULT_EXPERIMENT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "slm_update_experiments"
    / "fineweb_edu_canonical_slm_update_v1"
)
DEFAULT_TRAINING_CONFIG = Path(__file__).resolve().parent / "configs" / "slm_update_qwen25_0p5b_experiment.json"
EQUAL_BUDGET_ARMS = (
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_metadata(tokenizer: Any) -> Dict[str, Any]:
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "model_max_length": int(getattr(tokenizer, "model_max_length", 0) or 0),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
    }


def _count_arm_tokens(path: Path, tokenizer: Any, *, eos_per_record: bool = True) -> Dict[str, Any]:
    records = 0
    nonempty_records = 0
    tokens = 0
    words = 0
    max_record_tokens = 0
    eos_extra = 1 if eos_per_record and getattr(tokenizer, "eos_token_id", None) is not None else 0
    for record in iter_jsonl_records_resilient(path):
        records += 1
        text = str(record.get("text") or "")
        if not text.strip():
            continue
        nonempty_records += 1
        words += int(record.get("word_count") or len(text.split()))
        token_count = len(tokenizer(text, add_special_tokens=False).input_ids) + eos_extra
        tokens += token_count
        max_record_tokens = max(max_record_tokens, token_count)
    return {
        "path": str(path),
        "records": int(records),
        "nonempty_records": int(nonempty_records),
        "word_count": int(words),
        "token_count": int(tokens),
        "max_record_tokens": int(max_record_tokens),
        "sha256": _file_sha256(path),
    }


def freeze_plan(
    *,
    experiment_manifest_path: Path,
    training_config_path: Path,
    output_path: Path,
    local_files_only: bool,
) -> Dict[str, Any]:
    manifest = load_json(experiment_manifest_path)
    config = load_json(training_config_path)
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("transformers is required to freeze tokenizer budgets") from exc

    tokenizer_id = str((config.get("target_model") or {}).get("tokenizer_id") or "")
    if not tokenizer_id:
        raise RuntimeError(f"Missing target_model.tokenizer_id in {training_config_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=local_files_only, use_fast=True)
    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    arm_token_counts: Dict[str, Any] = {}
    missing_paths = []
    for arm_name in EQUAL_BUDGET_ARMS:
        path = Path(str((arms.get(arm_name) or {}).get("path") or ""))
        if not path.exists():
            missing_paths.append({"arm": arm_name, "path": str(path)})
            continue
        arm_token_counts[arm_name] = _count_arm_tokens(path, tokenizer)
    if missing_paths:
        raise FileNotFoundError(f"Missing equal-budget arm files: {missing_paths}")
    matched_budget = min(int(payload["token_count"]) for payload in arm_token_counts.values())
    primary_budget = min(
        int(arm_token_counts["curated_equal_budget"]["token_count"]),
        int(arm_token_counts["stageA_random_equal_budget"]["token_count"]),
    )
    sequence_length = int((config.get("token_budget") or {}).get("sequence_length") or 1024)
    plan = {
        "schema_version": "slm-update-frozen-plan-v1",
        "experiment_name": manifest.get("experiment_name"),
        "dataset": manifest.get("dataset"),
        "profile": manifest.get("profile"),
        "experiment_manifest_path": str(experiment_manifest_path),
        "training_config_path": str(training_config_path),
        "training_config_sha256": _file_sha256(training_config_path),
        "primary_comparison": "curated_equal_budget_vs_stageA_random_equal_budget",
        "framework_scope": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
        },
        "target_model": config.get("target_model"),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "token_budget": {
            "unit": "target_tokenizer_tokens_with_eos_per_record",
            "primary_matched_token_budget": int(primary_budget),
            "all_equal_budget_arms_matched_token_budget": int(matched_budget),
            "sequence_length": int(sequence_length),
            "estimated_primary_sequences": int(primary_budget // sequence_length),
            "estimated_all_arm_sequences": int(matched_budget // sequence_length),
            "packing_policy": (config.get("token_budget") or {}).get("truncate_or_pack_policy"),
            "overflow_policy": (config.get("token_budget") or {}).get("overflow_policy"),
        },
        "arm_token_counts": arm_token_counts,
        "training_recipe": config.get("training_recipe"),
        "evaluation_contract": config.get("evaluation_contract"),
        "contamination_controls": config.get("contamination_controls"),
        "required_training_runs": {
            "base_no_update": "evaluate only",
            "primary_train_arms": [
                "curated_equal_budget",
                "stageA_random_equal_budget",
            ],
            "supporting_train_arms": [
                "raw_random_equal_budget",
            ],
            "reference_arms": [
                "stageA_all_reference",
                "raw_all_reference",
            ],
            "seeds": [
                20260608,
                20260609,
                20260610,
            ],
        },
        "claim_boundary": {
            "allowed_if_successful": "Scoped target-SLM continued-pretraining benefit over equal-budget Stage-A random usable data.",
            "not_allowed": [
                "Universal curation improvement claim",
                "Using target-SLM outcomes as Stage-B selector objective",
                "Claiming raw-corpus success without a raw/provenance-rich G3 corpus",
            ],
        },
    }
    save_json(output_path, plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze tokenizer-budget SLM update plan.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    experiment_manifest = args.experiment_manifest or (args.experiment_dir / "manifest.json")
    output_path = args.output or (args.experiment_dir / "frozen_training_plan.json")
    plan = freeze_plan(
        experiment_manifest_path=experiment_manifest,
        training_config_path=args.training_config,
        output_path=output_path,
        local_files_only=not bool(args.allow_download),
    )
    print(
        {
            "experiment_name": plan["experiment_name"],
            "target_model": (plan["target_model"] or {}).get("model_id"),
            "primary_comparison": plan["primary_comparison"],
            "token_budget": plan["token_budget"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
