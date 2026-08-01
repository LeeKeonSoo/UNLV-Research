#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
PLAN_PATH = Path("configs") / "math_domain_natural_budget_v3_protocol_qwen3_4b.json"
OUTPUT_DIR_V3 = OUTPUT_DIR / "math_domain_natural_budget_v3_qwen3_4b"
BLOCKS_DIR = OUTPUT_DIR_V3 / "token_blocks"
VALIDATION_REPORT = OUTPUT_DIR / "validation" / "math_domain_natural_budget_v3_blocks_report.json"
TRAINING_ARMS = ("raw_full_natural", "curated_math_v2_natural", "curated_math_v3_natural")


@dataclass(frozen=True, slots=True)
class BlockBuildSpec:
    jsonl_path: Path
    output_path: Path
    sequence_length: int
    tokenizer: Any


def _load_tokenizer(plan: JsonMap) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        plan["target_model"]["tokenizer_id"],
        revision=plan["target_model"].get("revision", "main"),
        local_files_only=True,
        use_fast=True,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _token_stream(path: Path, tokenizer: Any) -> Iterable[int]:
    eos = tokenizer.eos_token_id
    for row in iter_jsonl_records_resilient(path):
        text = str(row.get("text") or "")
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if eos is not None:
            ids.append(int(eos))
        for token_id in ids:
            yield int(token_id)


def _build_blocks(spec: BlockBuildSpec) -> JsonMap:
    started = time.time()
    buffer: list[int] = []
    blocks: list[torch.Tensor] = []
    consumed = 0
    for token_id in _token_stream(spec.jsonl_path, spec.tokenizer):
        buffer.append(token_id)
        consumed += 1
        if len(buffer) == spec.sequence_length:
            blocks.append(torch.tensor(buffer, dtype=torch.int32))
            buffer = []
    if not blocks:
        raise RuntimeError(f"No complete blocks for {spec.jsonl_path}")
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.stack(blocks)
    torch.save({"input_ids": tensor}, spec.output_path)
    return {
        "source_jsonl": str(spec.jsonl_path),
        "source_sha256": sha256_file(spec.jsonl_path),
        "path": str(spec.output_path),
        "sha256": sha256_file(spec.output_path),
        "sequence_length": spec.sequence_length,
        "blocks": int(tensor.shape[0]),
        "tokens_in_blocks": int(tensor.numel()),
        "consumed_stream_tokens": int(consumed),
        "dropped_tail_tokens": int(consumed - tensor.numel()),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _arm_path(plan: JsonMap, arm: str) -> Path:
    row = plan["arm_token_counts"][arm]
    return Path(str(row["path"]))


def _update_plan(plan: JsonMap, manifest_path: Path, report: JsonMap) -> JsonMap:
    recipe = plan["confirmatory_training_recipe"]
    grad_accum = int(recipe["gradient_accumulation_steps"])
    recipe["optimizer_steps_by_arm"] = {
        arm: math.ceil(int(row["blocks"]) / grad_accum) for arm, row in report["blocks"].items()
    }
    recipe["natural_budget_packed_tokens_by_arm"] = {
        arm: int(row["tokens_in_blocks"]) for arm, row in report["blocks"].items()
    }
    recipe["natural_budget_blocks_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    recipe["same_step_count_for_every_arm"] = False
    plan["status"] = "math_selector_v3_natural_budget_blocks_frozen_before_training_outcomes"
    plan["stage_c_outcomes_read"] = False
    save_json(PLAN_PATH, plan)
    return plan


def prepare() -> JsonMap:
    plan = load_json(PLAN_PATH)
    recipe = plan["confirmatory_training_recipe"]
    sequence_length = int(recipe["sequence_length"])
    tokenizer = _load_tokenizer(plan)
    blocks = {
        arm: _build_blocks(
            BlockBuildSpec(
                jsonl_path=_arm_path(plan, arm),
                output_path=BLOCKS_DIR / f"{arm}.pt",
                sequence_length=sequence_length,
                tokenizer=tokenizer,
            )
        )
        for arm in TRAINING_ARMS
    }
    report = {
        "schema_version": "math-domain-natural-budget-v3-blocks-report-v1",
        "status": "math_natural_budget_v3_blocks_frozen",
        "plan": str(PLAN_PATH),
        "blocks": blocks,
        "sequence_length": sequence_length,
        "gradient_accumulation_steps": int(recipe["gradient_accumulation_steps"]),
        "stage_c_outcomes_read": False,
        "utility_scope": plan["utility_scope"],
        "claim_boundary": "Math selector v3 token-block freeze only; no training or success claim.",
    }
    manifest_path = BLOCKS_DIR / "block_manifest.json"
    save_json(manifest_path, report)
    report["sha256"] = sha256_file(manifest_path)
    plan = _update_plan(plan, manifest_path, report)
    report["plan_sha256"] = sha256_file(PLAN_PATH)
    save_json(VALIDATION_REPORT, report)
    return report


def main() -> int:
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
