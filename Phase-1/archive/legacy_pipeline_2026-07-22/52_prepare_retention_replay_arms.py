#!/usr/bin/env python3
"""Prepare retention-aware target/replay training-construction arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path("configs") / "retention_replay_development_plan_qwen25_0p5b_fineweb.json"
DEFAULT_REPLAY_BATCH_DIR = Path("validation") / "fixtures" / "wikitext103_subset"


def _arm_name(target_fraction: float, overrides: Dict[str, str] | None = None) -> str:
    override = (overrides or {}).get(str(target_fraction))
    if override:
        return str(override)
    percent = target_fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"retention_replay_target{int(round(percent)):03d}"
    basis_points = int(round(target_fraction * 10000.0))
    return f"retention_replay_target{basis_points:05d}"


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _text_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_count(text: str, tokenizer: Any) -> int:
    if not text.strip():
        return 0
    count = len(tokenizer(text, add_special_tokens=False).input_ids)
    return count + (1 if getattr(tokenizer, "eos_token_id", None) is not None else 0)


def _iter_replay_train(batch_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(batch_dir.glob("batch_*.json")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        for record in payload:
            if not isinstance(record, dict) or str(record.get("source_split") or "") != "train":
                continue
            text = str(record.get("text") or "")
            if text.strip():
                yield {
                    "id": str(record.get("id") or _text_hash(text)),
                    "text": text,
                    "source": "wikitext103_train_replay",
                    "source_split": "train",
                    "mixture_component": "general_replay",
                }


def _stable_replay_records(batch_dir: Path, seed: int) -> List[Dict[str, Any]]:
    records = list(_iter_replay_train(batch_dir))
    records.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest()
    )
    return records


def _take_to_budget(
    records: Iterable[Dict[str, Any]], tokenizer: Any, budget: int, component: str
) -> Tuple[List[Tuple[Dict[str, Any], int]], int]:
    if budget <= 0:
        return [], 0
    selected: List[Tuple[Dict[str, Any], int]] = []
    tokens = 0
    for record in records:
        text = str(record.get("text") or "")
        count = _token_count(text, tokenizer)
        if count <= 0:
            continue
        payload = dict(record)
        payload["mixture_component"] = component
        selected.append((payload, count))
        tokens += count
        if tokens >= budget:
            break
    return selected, tokens


def _interleave(
    target: List[Tuple[Dict[str, Any], int]],
    replay: List[Tuple[Dict[str, Any], int]],
    target_fraction: float,
) -> List[Dict[str, Any]]:
    target_idx = replay_idx = target_tokens = replay_tokens = 0
    merged: List[Dict[str, Any]] = []
    while target_idx < len(target) or replay_idx < len(replay):
        total = target_tokens + replay_tokens
        current_fraction = target_tokens / total if total else 0.0
        if current_fraction < target_fraction and target_idx < len(target):
            record, tokens = target[target_idx]
            target_idx += 1
            target_tokens += tokens
        elif replay_idx < len(replay):
            record, tokens = replay[replay_idx]
            replay_idx += 1
            replay_tokens += tokens
        elif target_idx < len(target):
            record, tokens = target[target_idx]
            target_idx += 1
            target_tokens += tokens
        else:
            break
        merged.append(record)
    return merged


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_arms(experiment_dir: Path, plan_path: Path, replay_batch_dir: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    frozen = load_json(experiment_dir / "frozen_training_plan.json")
    target_budget = int((frozen.get("token_budget") or {}).get("all_equal_budget_arms_matched_token_budget") or 0)
    model_id = str((frozen.get("target_model") or {}).get("model_id") or "")
    tokenizer = _load_tokenizer(model_id)
    target_path = experiment_dir / f"{plan['target_component']}.jsonl"
    target_records = list(iter_jsonl_records_resilient(target_path))
    replay_records = _stable_replay_records(replay_batch_dir, int(plan["replay_selection_seed"]))
    external_manifest = load_json(experiment_dir / "external_guardrails" / "external_guardrail_holdout_manifest.json")
    external_hashes = {
        _text_hash(str(record.get("text") or ""))
        for record in iter_jsonl_records_resilient(Path(external_manifest["output_path"]))
    }

    arms: Dict[str, Any] = {}
    for fraction in plan["candidate_target_fractions"]:
        target_fraction = float(fraction)
        replay_fraction = 1.0 - target_fraction
        arm_name = _arm_name(target_fraction, plan.get("arm_name_overrides"))
        target_selected, target_tokens = _take_to_budget(
            target_records, tokenizer, int(target_budget * target_fraction), "coverage_backfilled_target"
        )
        replay_selected, replay_tokens = _take_to_budget(
            replay_records, tokenizer, int(target_budget * replay_fraction), "general_replay"
        )
        replay_overlap = sum(
            1 for record, _tokens in replay_selected if _text_hash(str(record.get("text") or "")) in external_hashes
        )
        if replay_overlap:
            raise RuntimeError(f"Replay/external exact overlap detected for {arm_name}: {replay_overlap}")
        merged = _interleave(target_selected, replay_selected, target_fraction)
        output_path = experiment_dir / f"{arm_name}.jsonl"
        record_count = _write_jsonl(output_path, merged)
        arms[arm_name] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "records": record_count,
            "target_fraction": target_fraction,
            "replay_fraction": replay_fraction,
            "target_tokens": target_tokens,
            "replay_tokens": replay_tokens,
            "replay_external_exact_overlap": replay_overlap,
        }

    manifest = {
        "schema_version": "retention-replay-arms-v1",
        "plan": plan,
        "target_token_budget": target_budget,
        "target_source": str(target_path),
        "replay_source": str(replay_batch_dir.resolve()),
        "arms": arms,
        "framework_scope": {
            "stage_b": "unchanged; no Utility or target-model outcomes",
            "stage_c": "evaluate target and retention outcomes",
            "release_layer": "compare retention-aware training-construction candidates",
            "utility_scope": "Stage C validation only; never selector objective",
        },
    }
    safe_plan_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in str(plan.get("plan_name") or "plan")
    )
    save_json(experiment_dir / f"retention_replay_arms_manifest_{safe_plan_name}.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare retention replay arms.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--replay-batch-dir", type=Path, default=DEFAULT_REPLAY_BATCH_DIR)
    args = parser.parse_args()
    manifest = prepare_arms(args.experiment_dir, args.plan, args.replay_batch_dir)
    print(json.dumps(manifest["arms"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
