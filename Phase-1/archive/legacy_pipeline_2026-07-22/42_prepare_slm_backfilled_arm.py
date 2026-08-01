#!/usr/bin/env python3
"""Prepare exploratory coverage-backfilled SLM training arm.

This is a release/training-construction candidate, not a selector objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_OUTPUT_NAME = "coverage_backfilled_interleaved50_equal_budget.jsonl"


def _uid(record: Dict[str, Any]) -> str:
    return str(record.get("chunk_uid") or record.get("id") or record.get("doc_id") or "")


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _record_tokens(record: Dict[str, Any], tokenizer: Any) -> int:
    text = str(record.get("text") or "")
    if not text.strip():
        return 0
    count = len(tokenizer(text, add_special_tokens=False).input_ids)
    return count + (1 if getattr(tokenizer, "eos_token_id", None) is not None else 0)


def _take_to_budget(
    records: Iterable[Dict[str, Any]],
    tokenizer: Any,
    *,
    budget_tokens: int,
    arm_label: str,
    excluded_uids: set[str] | None = None,
) -> Tuple[List[Tuple[Dict[str, Any], int]], int]:
    selected: List[Tuple[Dict[str, Any], int]] = []
    tokens = 0
    excluded = excluded_uids or set()
    for record in records:
        uid = _uid(record)
        if uid in excluded:
            continue
        token_count = _record_tokens(record, tokenizer)
        if token_count <= 0:
            continue
        payload = dict(record)
        payload["arm"] = arm_label
        payload["mixture_component"] = arm_label
        selected.append((payload, token_count))
        tokens += token_count
        if tokens >= budget_tokens:
            break
    return selected, tokens


def _interleave_by_token_share(
    core_records: List[Tuple[Dict[str, Any], int]],
    support_records: List[Tuple[Dict[str, Any], int]],
    *,
    core_fraction: float,
) -> List[Dict[str, Any]]:
    core_idx = 0
    support_idx = 0
    core_tokens = 0
    support_tokens = 0
    merged: List[Dict[str, Any]] = []
    while core_idx < len(core_records) or support_idx < len(support_records):
        total = core_tokens + support_tokens
        current_core_fraction = (core_tokens / total) if total else 0.0
        take_core = current_core_fraction < core_fraction
        if take_core and core_idx < len(core_records):
            record, tokens = core_records[core_idx]
            merged.append(record)
            core_tokens += tokens
            core_idx += 1
        elif support_idx < len(support_records):
            record, tokens = support_records[support_idx]
            merged.append(record)
            support_tokens += tokens
            support_idx += 1
        elif core_idx < len(core_records):
            record, tokens = core_records[core_idx]
            merged.append(record)
            core_tokens += tokens
            core_idx += 1
        else:
            break
    return merged


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    words = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            try:
                words += int(record.get("word_count") or 0)
            except (TypeError, ValueError):
                words += 0
    return {"path": str(path), "records": int(count), "word_count": int(words)}


def build_backfilled_arm(
    *,
    experiment_dir: Path,
    output_name: str,
    selected_core_fraction: float,
) -> Dict[str, Any]:
    manifest = load_json(experiment_dir / "manifest.json")
    frozen = load_json(experiment_dir / "frozen_training_plan.json")
    model_id = str((frozen.get("target_model") or {}).get("model_id") or "")
    tokenizer = _load_tokenizer(model_id)
    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    curated_path = Path(str((arms.get("curated_equal_budget") or {}).get("path") or ""))
    stagea_path = Path(str((arms.get("stageA_random_equal_budget") or {}).get("path") or ""))
    if not curated_path.exists() or not stagea_path.exists():
        raise FileNotFoundError({"curated": str(curated_path), "stageA_random": str(stagea_path)})
    target_tokens = int(((frozen.get("token_budget") or {}).get("all_equal_budget_arms_matched_token_budget") or 0))
    if target_tokens <= 0:
        raise RuntimeError("Missing positive target token budget")
    core_budget = int(target_tokens * float(selected_core_fraction))
    support_budget = max(0, target_tokens - core_budget)
    curated_records = list(iter_jsonl_records_resilient(curated_path))
    stagea_records = list(iter_jsonl_records_resilient(stagea_path))
    core_records, core_tokens = _take_to_budget(
        curated_records,
        tokenizer,
        budget_tokens=core_budget,
        arm_label="selected_core",
    )
    core_uids = {_uid(record) for record, _tokens in core_records}
    support_records, support_tokens = _take_to_budget(
        stagea_records,
        tokenizer,
        budget_tokens=support_budget,
        arm_label="stageA_coverage_backfill",
        excluded_uids=core_uids,
    )
    output_path = experiment_dir / output_name
    merged_records = _interleave_by_token_share(
        core_records,
        support_records,
        core_fraction=float(selected_core_fraction),
    )
    write_summary = _write_jsonl(output_path, merged_records)
    summary = {
        "schema_version": "slm-backfilled-arm-v1",
        "scope": "exploratory_training_construction_candidate_not_selector_objective",
        "utility_scope": "Stage C validation only; never selector objective",
        "arm_name": output_path.stem,
        "path": str(output_path),
        "model_id": model_id,
        "target_tokens": int(target_tokens),
        "selected_core_fraction": float(selected_core_fraction),
        "components": {
            "selected_core": {
                "records": len(core_records),
                "tokens": int(core_tokens),
                "requested_tokens": int(core_budget),
            },
            "stageA_coverage_backfill": {
                "records": len(support_records),
                "tokens": int(support_tokens),
                "requested_tokens": int(support_budget),
            },
        },
        "write_summary": write_summary,
        "ordering_policy": "interleave_components_to_preserve_selected_core_fraction_through_prefixes",
        "rationale": "Selected-only full-budget training underperformed Stage-A random on the first certification seed; this candidate preserves selected high-quality core while adding broad Stage-A support.",
        "claim_boundary": "Exploratory candidate only; requires predeclared validation before any certification claim.",
    }
    save_json(experiment_dir / f"{output_path.stem}_manifest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare coverage-backfilled SLM arm.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--selected-core-fraction", type=float, default=0.5)
    args = parser.parse_args()
    summary = build_backfilled_arm(
        experiment_dir=args.experiment_dir,
        output_name=str(args.output_name),
        selected_core_fraction=float(args.selected_core_fraction),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
