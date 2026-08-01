#!/usr/bin/env python3
"""Prepare disjoint held-out eval records for the frozen SLM update experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import INDEX_DIR, OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import SCORING_MANIFEST_PATH, _passes_gates, _stage_a_gate


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "configs" / "curation_profiles.json"
DEFAULT_INDEX_DB_PATH = INDEX_DIR / "index.sqlite"


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


def _stable_score(value: str, *, seed: int, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _load_index_texts(index_db_path: Path, uids: Iterable[str]) -> Dict[str, str]:
    unique_uids = sorted({str(uid) for uid in uids if str(uid)})
    if not index_db_path.exists():
        raise FileNotFoundError(f"Missing index DB needed to materialize eval text: {index_db_path}")
    texts: Dict[str, str] = {}
    conn = sqlite3.connect(str(index_db_path))
    try:
        for start in range(0, len(unique_uids), 900):
            batch = unique_uids[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            for chunk_uid, text in conn.execute(
                f"SELECT chunk_uid, text FROM chunks WHERE chunk_uid IN ({placeholders})",
                batch,
            ):
                texts[str(chunk_uid)] = str(text or "")
    finally:
        conn.close()
    return texts


def _training_arm_uids(arms: Dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for arm_name in ("curated_equal_budget", "stageA_random_equal_budget", "raw_random_equal_budget"):
        path = Path(str((arms.get(arm_name) or {}).get("path") or ""))
        if not path.exists():
            raise FileNotFoundError(f"Missing training arm file for disjoint holdout construction: {path}")
        for record in iter_jsonl_records_resilient(path):
            uid = _uid(record)
            if uid:
                used.add(uid)
    return used


def _profile_payload(path: Path, profile_name: str) -> Dict[str, Any]:
    profile = ((load_json(path).get("profiles") or {}).get(str(profile_name)) or {})
    if not isinstance(profile, dict) or not profile:
        raise RuntimeError(f"Profile {profile_name!r} not found in {path}")
    return profile


def _scored_path(dataset: str) -> Path:
    manifest = load_json(SCORING_MANIFEST_PATH)
    path = (((manifest.get("datasets") or {}).get(dataset) or {}).get("path"))
    if not path:
        raise RuntimeError(f"Dataset {dataset!r} not found in scoring manifest")
    return Path(str(path))


def _minimal_eval_record(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    return {
        "id": _uid(record),
        "text": text,
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


def build_holdout(
    *,
    experiment_dir: Path,
    profiles_path: Path,
    index_db_path: Path,
    seed: int,
    target_word_budget: int,
    output_name: str,
) -> Dict[str, Any]:
    experiment_manifest_path = experiment_dir / "manifest.json"
    manifest = load_json(experiment_manifest_path)
    dataset = str(manifest.get("dataset") or "")
    profile = str(manifest.get("profile") or "")
    profile_cfg = _profile_payload(profiles_path, profile)
    stage_a = _stage_a_gate(profile_cfg)
    scored_path = _scored_path(dataset)
    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    training_uids = _training_arm_uids(arms)
    scored_records = list(iter_jsonl_records_resilient(scored_path))
    candidates = [
        record
        for record in scored_records
        if _uid(record) not in training_uids and _passes_gates(record, stage_a)
    ]
    candidates.sort(key=lambda record: (_stable_score(_uid(record), seed=seed, salt="slm-eval-holdout"), _uid(record)))

    selected: List[Dict[str, Any]] = []
    words = 0
    for record in candidates:
        if words >= target_word_budget:
            break
        selected.append(record)
        words += _word_count(record)
    text_by_uid = _load_index_texts(index_db_path, (_uid(record) for record in selected))
    output_path = experiment_dir / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    written_words = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            text = text_by_uid.get(_uid(record), "")
            if not text.strip():
                continue
            payload = _minimal_eval_record(record, text)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            written_words += int(payload.get("word_count") or 0)

    holdout_manifest = {
        "schema_version": "slm-update-eval-holdout-v1",
        "experiment_name": manifest.get("experiment_name"),
        "dataset": dataset,
        "profile": profile,
        "seed": int(seed),
        "source": "Stage-A records disjoint from curated, Stage-A-random, and raw-random training arms.",
        "claim_scope": "internal held-out same-corpus distribution; final paper claims still require contamination and external benchmark checks",
        "framework_scope": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
        },
        "paths": {
            "experiment_manifest": str(experiment_manifest_path),
            "scored_path": str(scored_path),
            "eval_jsonl": str(output_path),
        },
        "counts": {
            "training_arm_uids": len(training_uids),
            "candidate_stage_a_records_excluding_training_arms": len(candidates),
            "target_word_budget": int(target_word_budget),
            "written_records": int(written),
            "written_word_count": int(written_words),
        },
    }
    save_json(experiment_dir / "eval_holdout_manifest.json", holdout_manifest)
    return holdout_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare disjoint SLM eval holdout.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB_PATH)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--target-word-budget", type=int, default=1000000)
    parser.add_argument("--output-name", default="heldout_stageA_eval.jsonl")
    args = parser.parse_args()
    manifest = build_holdout(
        experiment_dir=args.experiment_dir,
        profiles_path=args.profiles,
        index_db_path=args.index_db,
        seed=int(args.seed),
        target_word_budget=int(args.target_word_budget),
        output_name=str(args.output_name),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
