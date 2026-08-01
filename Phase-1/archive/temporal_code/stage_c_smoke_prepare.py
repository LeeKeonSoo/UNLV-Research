#!/usr/bin/env python3
"""Construct and freeze target-token temporal-code Stage-C smoke arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


DEFAULT_CONTRACT = Path("configs") / "temporal_code_stage_c_smoke_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_stage_c_smoke_qwen3_4b_v1"


def _uid(row: Dict[str, Any]) -> str:
    return str(row.get("chunk_uid") or row.get("id") or row.get("record_id") or "")


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return [row for row in iter_jsonl_records_resilient(path) if str(row.get("text") or "").strip()]


def _token_count(row: Dict[str, Any], tokenizer: Any) -> int:
    return len(tokenizer(str(row["text"]), add_special_tokens=False).input_ids) + 1


def _stable_random(rows: Iterable[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    ranked = []
    for row in rows:
        identity = _uid(row)
        digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8", errors="replace")).hexdigest()
        ranked.append((digest, identity, row))
    return [row for _, _, row in sorted(ranked)]


def _take_to_budget(rows: Iterable[Dict[str, Any]], tokenizer: Any, budget: int) -> List[Dict[str, Any]]:
    selected = []
    total = 0
    for row in rows:
        count = _token_count(row, tokenizer)
        if total + count > budget:
            continue
        payload = dict(row)
        payload["target_token_count_with_eos"] = count
        selected.append(payload)
        total += count
        if total == budget:
            break
    return selected


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    tokens = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            records += 1
            tokens += int(row["target_token_count_with_eos"])
    return {"path": str(path), "records": records, "target_tokens_with_eos": tokens, "sha256": sha256_file(path)}


def prepare(contract_path: Path, output_dir: Path, *, allow_download: bool) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    contract = load_json(contract_path)
    tokenizer_id = contract["target_model"]["tokenizer_id"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=not allow_download, use_fast=True)
    paths = {name: Path(value) for name, value in contract["input_paths"].items()}
    curated = _load_rows(paths["curated"])
    baseline = _load_rows(paths["common_stage_a_random"])
    raw = _load_rows(paths["raw_stage0_train"])
    curated_ids = {_uid(row) for row in curated}
    baseline_ids = {_uid(row) for row in baseline}
    overlap = sorted(curated_ids.intersection(baseline_ids))
    if overlap:
        raise RuntimeError(f"Curated/common Stage-A baseline overlap: {overlap[:5]}")

    # Raw-random is a supporting pre-Stage-A record-level baseline. It is not
    # used as the common sensitivity baseline and cannot tune Stage B.
    raw_ranked = _stable_random(raw, int(contract["arm_contract"]["raw_random_seed"]))
    available = {
        "curated_equal_token": sum(_token_count(row, tokenizer) for row in curated),
        "stageA_random_equal_token": sum(_token_count(row, tokenizer) for row in baseline),
        "raw_random_equal_token": sum(_token_count(row, tokenizer) for row in raw_ranked),
    }
    budget = min(int(contract["arm_contract"]["smoke_max_tokens"]), *available.values())
    arm_rows = {
        "curated_equal_token": _take_to_budget(curated, tokenizer, budget),
        "stageA_random_equal_token": _take_to_budget(baseline, tokenizer, budget),
        "raw_random_equal_token": _take_to_budget(raw_ranked, tokenizer, budget),
    }
    materialized = {
        name: _write_jsonl(output_dir / f"{name}.jsonl", rows)
        for name, rows in arm_rows.items()
    }
    realized = [row["target_tokens_with_eos"] for row in materialized.values()]
    realized_budget = min(realized)
    if max(realized) - min(realized) > 2048:
        raise RuntimeError(f"Target-token arm mismatch exceeds one sequence: {realized}")
    common_baseline_identity = materialized["stageA_random_equal_token"]["sha256"]
    sensitivity_common_baseline = {
        name: common_baseline_identity
        for name in ("curated_equal_token", "raw_random_equal_token")
    }
    report = {
        "schema_version": "temporal-code-stage-c-smoke-arms-v1",
        "status": "frozen_target_token_arms_before_model_execution",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "target_model": contract["target_model"],
        "tokenizer": {
            "name_or_path": str(tokenizer.name_or_path),
            "vocab_size": int(tokenizer.vocab_size),
            "eos_token_id": tokenizer.eos_token_id,
        },
        "available_target_tokens_with_eos": available,
        "matched_budget_target_tokens_with_eos": realized_budget,
        "arms": materialized,
        "curated_common_baseline_overlap_count": 0,
        "common_stage_a_baseline_sha256": common_baseline_identity,
        "sensitivity_common_stage_a_baseline_sha256": sensitivity_common_baseline,
        "all_sensitivity_arms_share_common_stage_a_baseline": len(set(sensitivity_common_baseline.values())) == 1,
        "confirmatory_content_read": False,
        "utility_scope": contract["utility_scope"],
        "claim_boundary": "Target-token arm construction only; no Utility or training-benefit claim.",
    }
    save_json(output_dir / "frozen_smoke_arm_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare temporal-code Qwen3-4B Stage-C smoke arms.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = prepare(args.contract, args.output_dir, allow_download=bool(args.allow_download))
    print(
        {
            "status": report["status"],
            "matched_budget_target_tokens_with_eos": report["matched_budget_target_tokens_with_eos"],
            "arms": report["arms"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
