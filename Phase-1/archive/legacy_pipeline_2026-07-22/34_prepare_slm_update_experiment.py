#!/usr/bin/env python3
"""Prepare frozen SLM-update experiment arms from current curation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import INDEX_DIR, OUTPUT_DIR, RUN_SUMMARY_PATH, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import SCORING_MANIFEST_PATH, _passes_gates, _stage_a_gate


DEFAULT_OUTPUT_ROOT = OUTPUT_DIR / "slm_update_experiments"
DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "configs" / "curation_profiles.json"
DEFAULT_INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
DEFAULT_ARMS = (
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
)


def _stable_score(value: str, *, seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _uid(record: Dict[str, Any]) -> str:
    diagnostics = record.get("diagnostics") if isinstance(record.get("diagnostics"), dict) else {}
    text = str(record.get("text") or "")
    return str(
        record.get("chunk_uid")
        or record.get("id")
        or diagnostics.get("text_hash")
        or (hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest() if text else "")
        or record.get("doc_id")
        or ""
    )


def _word_count(record: Dict[str, Any]) -> int:
    try:
        value = int(record.get("word_count") or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else len(str(record.get("text") or "").split())


def _profile_payload(path: Path, profile_name: str) -> Dict[str, Any]:
    profile = ((load_json(path).get("profiles") or {}).get(str(profile_name)) or {})
    if not isinstance(profile, dict) or not profile:
        raise RuntimeError(f"Profile {profile_name!r} not found in {path}")
    return profile


def _selected_path(dataset: str, profile: str) -> Path:
    run_summary = load_json(RUN_SUMMARY_PATH) if RUN_SUMMARY_PATH.exists() else {}
    summary_path = ((((run_summary.get("profiles") or {}).get(profile) or {}).get(dataset) or {}).get("output_path"))
    if summary_path:
        path = Path(str(summary_path))
        if path.exists():
            return path
    return OUTPUT_DIR / "subsets" / profile / f"{dataset}.jsonl"


def _scored_path(dataset: str) -> Path:
    manifest = load_json(SCORING_MANIFEST_PATH)
    path = (((manifest.get("datasets") or {}).get(dataset) or {}).get("path"))
    if not path:
        raise RuntimeError(f"Dataset {dataset!r} not found in scoring manifest")
    return Path(str(path))


def _minimal_training_record(record: Dict[str, Any], *, arm: str) -> Dict[str, Any]:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    return {
        "id": _uid(record),
        "text": str(record.get("text") or ""),
        "arm": arm,
        "source": record.get("source"),
        "doc_id": record.get("doc_id"),
        "chunk_id": record.get("chunk_id"),
        "chunk_uid": record.get("chunk_uid"),
        "word_count": _word_count(record),
        "provenance": {
            "input_source": provenance.get("input_source"),
            "metadata": provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {},
        },
    }


def _select_to_budget(records: Iterable[Dict[str, Any]], *, budget_words: int, seed: int, salt: str) -> List[Dict[str, Any]]:
    candidates = [record for record in records if _word_count(record) > 0 and str(record.get("text") or "").strip()]
    candidates.sort(key=lambda record: (_stable_score(_uid(record), seed=seed, salt=salt), _uid(record)))
    selected: List[Dict[str, Any]] = []
    total_words = 0
    for record in candidates:
        if total_words >= budget_words:
            break
        selected.append(record)
        total_words += _word_count(record)
    return selected


def _summarize_records(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    count = 0
    words = 0
    sources: Dict[str, int] = {}
    for record in records:
        count += 1
        words += _word_count(record)
        source = str(record.get("source") or "unknown")
        sources[source] = sources.get(source, 0) + 1
    return {
        "records": int(count),
        "word_count": int(words),
        "top_sources": dict(sorted(sources.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }


def _load_index_texts(index_db_path: Path, uids: Iterable[str]) -> Dict[str, str]:
    unique_uids = sorted({str(uid) for uid in uids if str(uid)})
    if not unique_uids:
        return {}
    if not index_db_path.exists():
        raise FileNotFoundError(f"Missing index DB needed to materialize training text: {index_db_path}")
    texts: Dict[str, str] = {}
    conn = sqlite3.connect(str(index_db_path))
    try:
        for start in range(0, len(unique_uids), 900):
            batch = unique_uids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            cursor = conn.execute(
                f"SELECT chunk_uid, text FROM chunks WHERE chunk_uid IN ({placeholders})",
                batch,
            )
            for chunk_uid, text in cursor:
                texts[str(chunk_uid)] = str(text or "")
    finally:
        conn.close()
    return texts


def _attach_texts(records: Iterable[Dict[str, Any]], text_by_uid: Dict[str, str]) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for record in records:
        if str(record.get("text") or "").strip():
            hydrated.append(record)
            continue
        uid = _uid(record)
        text = text_by_uid.get(uid, "")
        if not text.strip():
            continue
        payload = dict(record)
        payload["text"] = text
        hydrated.append(payload)
    return hydrated


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]], *, arm: str) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    words = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = _minimal_training_record(record, arm=arm)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            words += int(payload.get("word_count") or 0)
    return {"path": str(path), "records": int(count), "word_count": int(words)}


def build_experiment(
    *,
    dataset: str,
    profile: str,
    profiles_path: Path,
    index_db_path: Path,
    experiment_name: str,
    output_root: Path,
    seed: int,
    token_budget_words: int | None,
    include_reference_all: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    profile_cfg = _profile_payload(profiles_path, profile)
    stage_a = _stage_a_gate(profile_cfg)
    selected_path = _selected_path(dataset, profile)
    scored_path = _scored_path(dataset)
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected subset: {selected_path}")
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored records: {scored_path}")

    selected_records = list(iter_jsonl_records_resilient(selected_path))
    selected_uids = {_uid(record) for record in selected_records}
    scored_records = list(iter_jsonl_records_resilient(scored_path))
    stage_a_records = [record for record in scored_records if _passes_gates(record, stage_a)]
    stage_a_control_records = [record for record in stage_a_records if _uid(record) not in selected_uids]
    raw_control_records = [record for record in scored_records if _uid(record) not in selected_uids]
    curated_words = sum(_word_count(record) for record in selected_records)
    budget_words = int(token_budget_words or curated_words)
    if budget_words <= 0:
        raise RuntimeError("Token/word budget must be positive")
    text_uids = {_uid(record) for record in raw_control_records}
    if include_reference_all:
        text_uids.update(_uid(record) for record in scored_records)
    text_by_uid = _load_index_texts(index_db_path, text_uids)
    stage_a_control_records = _attach_texts(stage_a_control_records, text_by_uid)
    raw_control_records = _attach_texts(raw_control_records, text_by_uid)
    if include_reference_all:
        selected_by_uid = {_uid(record): record for record in selected_records if str(record.get("text") or "").strip()}
        scored_records = _attach_texts(
            (selected_by_uid.get(_uid(record), record) for record in scored_records),
            text_by_uid,
        )
        stage_a_records = _attach_texts(
            (selected_by_uid.get(_uid(record), record) for record in stage_a_records),
            text_by_uid,
        )

    arm_sources = {
        "curated_equal_budget": _select_to_budget(selected_records, budget_words=budget_words, seed=seed, salt="curated"),
        "stageA_random_equal_budget": _select_to_budget(stage_a_control_records, budget_words=budget_words, seed=seed, salt="stageA-random"),
        "raw_random_equal_budget": _select_to_budget(raw_control_records, budget_words=budget_words, seed=seed, salt="raw-random"),
    }
    reference_summaries = {
        "stageA_all_reference": _summarize_records(stage_a_records),
        "raw_all_reference": _summarize_records(scored_records),
    }
    out_dir = output_root / experiment_name
    arms: Dict[str, Any] = {}
    for arm_name in DEFAULT_ARMS:
        records = arm_sources[arm_name]
        arms[arm_name] = (
            {"path": None, **_summarize_records(records)}
            if dry_run
            else _write_jsonl(out_dir / f"{arm_name}.jsonl", records, arm=arm_name)
        )
    if include_reference_all:
        for arm_name, source_records in (("stageA_all_reference", stage_a_records), ("raw_all_reference", scored_records)):
            arms[arm_name] = (
                {"path": None, **reference_summaries[arm_name]}
                if dry_run
                else _write_jsonl(out_dir / f"{arm_name}.jsonl", source_records, arm=arm_name)
            )
    else:
        arms.update(reference_summaries)

    manifest = {
        "schema_version": "slm-update-experiment-v1",
        "experiment_name": experiment_name,
        "dataset": dataset,
        "profile": profile,
        "profiles_path": str(profiles_path),
        "index_db_path": str(index_db_path),
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "purpose": "Freeze equal-budget continued-training arms for target-SLM curation validation.",
        "primary_comparison": "curated_equal_budget_vs_stageA_random_equal_budget",
        "primary_success_criterion": "curated_equal_budget improves target-SLM evaluation over stageA_random_equal_budget at matched token/compute budget and replicated seeds.",
        "framework_scope": {
            "data_collection": "upstream",
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
        },
        "target_model": {
            "status": "not_selected",
            "selection_rule": "Choose and freeze a 2024-era small language model checkpoint before observing target-model results.",
        },
        "budget": {
            "unit": "word_count_proxy",
            "equal_budget_words": int(budget_words),
            "curated_full_words": int(curated_words),
        },
        "inputs": {
            "selected_path": str(selected_path),
            "scored_path": str(scored_path),
            "selected_records": len(selected_records),
            "scored_records": len(scored_records),
            "stage_a_records": len(stage_a_records),
            "stage_a_control_records_excluding_selected": len(stage_a_control_records),
            "raw_control_records_excluding_selected": len(raw_control_records),
        },
        "arms": arms,
        "required_training_runs": {
            "base_no_update": "evaluate only",
            "curated_equal_budget": "train with same tokens/compute as stageA_random_equal_budget",
            "stageA_random_equal_budget": "primary operational baseline",
            "raw_random_equal_budget": "raw/unfiltered stress baseline",
            "stageA_all_reference": "optional larger-budget reference",
            "raw_all_reference": "optional larger-budget reference",
            "min_primary_seeds": 3,
        },
        "required_evaluation": [
            "held_out_new_data_distribution",
            "general_capability_benchmarks",
            "forgetting_or_regression_suite",
            "benchmark_contamination_audit",
            "domain_source_slice_analysis",
            "training_stability_and_seed_variance",
            "cost_and_retained_token_efficiency",
        ],
    }
    if not dry_run:
        save_json(out_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SLM-update experiment arms.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB_PATH)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--token-budget-words", type=int)
    parser.add_argument("--include-reference-all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = build_experiment(
        dataset=str(args.dataset),
        profile=str(args.profile),
        profiles_path=args.profiles,
        index_db_path=args.index_db,
        experiment_name=str(args.experiment_name),
        output_root=args.output_root,
        seed=int(args.seed),
        token_budget_words=args.token_budget_words,
        include_reference_all=bool(args.include_reference_all),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps({
        "experiment_name": manifest["experiment_name"],
        "dataset": manifest["dataset"],
        "profile": manifest["profile"],
        "dry_run": manifest["dry_run"],
        "budget": manifest["budget"],
        "arms": manifest["arms"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
