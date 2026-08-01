#!/usr/bin/env python3
"""Prepare untouched, mutually disjoint holdouts for backfill confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import INDEX_DIR, OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import SCORING_MANIFEST_PATH, _passes_gates, _stage_a_gate


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "configs" / "curation_profiles.json"
DEFAULT_INDEX_DB_PATH = INDEX_DIR / "index.sqlite"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _stable_score(value: str, *, seed: int, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()


def _style_bucket(record: Dict[str, Any]) -> str:
    core = record.get("core_metrics") if isinstance(record.get("core_metrics"), dict) else {}
    validity = core.get("structural_validity_gate") if isinstance(core.get("structural_validity_gate"), dict) else {}
    details = validity.get("details") if isinstance(validity.get("details"), dict) else {}
    return str(details.get("style_bucket") or "unknown")


def _length_bucket(record: Dict[str, Any]) -> str:
    words = _word_count(record)
    if words < 64:
        return "short"
    if words < 256:
        return "medium"
    if words < 1024:
        return "long"
    return "very_long"


def _load_uids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {_uid(record) for record in iter_jsonl_records_resilient(path) if _uid(record)}


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


def _take_random(records: List[Dict[str, Any]], *, budget_words: int, seed: int, salt: str) -> List[Dict[str, Any]]:
    ordered = sorted(records, key=lambda record: (_stable_score(_uid(record), seed=seed, salt=salt), _uid(record)))
    selected: List[Dict[str, Any]] = []
    words = 0
    for record in ordered:
        if words >= budget_words:
            break
        selected.append(record)
        words += _word_count(record)
    return selected


def _take_stratified(records: List[Dict[str, Any]], *, budget_words: int, seed: int) -> List[Dict[str, Any]]:
    strata: Dict[str, deque[Dict[str, Any]]] = {}
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[f"{_style_bucket(record)}::{_length_bucket(record)}"].append(record)
    for key, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda record: (_stable_score(_uid(record), seed=seed, salt=f"confirmatory-stratum:{key}"), _uid(record)),
        )
        strata[key] = deque(ordered)
    selected: List[Dict[str, Any]] = []
    words = 0
    active = sorted(strata)
    while active and words < budget_words:
        next_active: List[str] = []
        for key in active:
            bucket = strata[key]
            if bucket and words < budget_words:
                record = bucket.popleft()
                selected.append(record)
                words += _word_count(record)
            if bucket:
                next_active.append(key)
        active = next_active
    return selected


def _write_eval(path: Path, records: List[Dict[str, Any]], text_by_uid: Dict[str, str], *, eval_slice: str) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    written_uids: List[str] = []
    words = 0
    styles: Dict[str, int] = defaultdict(int)
    lengths: Dict[str, int] = defaultdict(int)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            uid = _uid(record)
            text = text_by_uid.get(uid, "")
            if not text.strip():
                continue
            payload = {
                "id": uid,
                "text": text,
                "source": record.get("source"),
                "doc_id": record.get("doc_id"),
                "chunk_id": record.get("chunk_id"),
                "chunk_uid": record.get("chunk_uid"),
                "word_count": _word_count(record),
                "eval_slice": eval_slice,
                "style_bucket": _style_bucket(record),
                "length_bucket": _length_bucket(record),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written_uids.append(uid)
            words += int(payload["word_count"])
            styles[str(payload["style_bucket"])] += 1
            lengths[str(payload["length_bucket"])] += 1
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "uid_set_sha256": hashlib.sha256("\n".join(sorted(written_uids)).encode("utf-8")).hexdigest(),
        "records": len(written_uids),
        "word_count": int(words),
        "style_counts": dict(sorted(styles.items())),
        "length_counts": dict(sorted(lengths.items())),
    }


def build_holdouts(
    *,
    experiment_dir: Path,
    profiles_path: Path,
    index_db_path: Path,
    broad_seed: int,
    stratified_seed: int,
    broad_word_budget: int,
    stratified_word_budget: int,
) -> Dict[str, Any]:
    experiment_manifest = load_json(experiment_dir / "manifest.json")
    backfill_manifest = load_json(experiment_dir / "coverage_backfilled_interleaved50_equal_budget_manifest.json")
    dataset = str(experiment_manifest.get("dataset") or "")
    profile = str(experiment_manifest.get("profile") or "")
    profile_cfg = ((load_json(profiles_path).get("profiles") or {}).get(profile) or {})
    stage_a = _stage_a_gate(profile_cfg)
    scored_manifest = load_json(SCORING_MANIFEST_PATH)
    scored_path = Path(str((((scored_manifest.get("datasets") or {}).get(dataset) or {}).get("path") or "")))
    if not scored_path.exists():
        raise FileNotFoundError(scored_path)

    arms = experiment_manifest.get("arms") if isinstance(experiment_manifest.get("arms"), dict) else {}
    excluded_paths = [
        Path(str((arms.get("curated_equal_budget") or {}).get("path") or "")),
        Path(str((arms.get("stageA_random_equal_budget") or {}).get("path") or "")),
        Path(str((arms.get("raw_random_equal_budget") or {}).get("path") or "")),
        Path(str(backfill_manifest.get("path") or "")),
        experiment_dir / "heldout_stageA_eval.jsonl",
    ]
    excluded_uids: set[str] = set()
    excluded_sources: List[Dict[str, Any]] = []
    for path in excluded_paths:
        uids = _load_uids(path)
        excluded_uids.update(uids)
        excluded_sources.append({"path": str(path), "sha256": _file_sha256(path), "uids": len(uids)})

    scored_records = list(iter_jsonl_records_resilient(scored_path))
    candidates = [
        record
        for record in scored_records
        if _uid(record) and _uid(record) not in excluded_uids and _passes_gates(record, stage_a)
    ]
    broad = _take_random(candidates, budget_words=broad_word_budget, seed=broad_seed, salt="confirmatory-broad-stageA")
    broad_uids = {_uid(record) for record in broad}
    stratified_candidates = [record for record in candidates if _uid(record) not in broad_uids]
    stratified = _take_stratified(stratified_candidates, budget_words=stratified_word_budget, seed=stratified_seed)
    stratified_uids = {_uid(record) for record in stratified}
    overlap = broad_uids & stratified_uids
    if overlap:
        raise RuntimeError(f"Confirmatory holdout slices overlap: {len(overlap)} records")
    text_by_uid = _load_index_texts(index_db_path, broad_uids | stratified_uids)

    broad_summary = _write_eval(
        experiment_dir / "confirmatory_broad_stageA_eval.jsonl",
        broad,
        text_by_uid,
        eval_slice="confirmatory_broad_stageA_primary",
    )
    stratified_summary = _write_eval(
        experiment_dir / "confirmatory_coverage_stratified_stageA_eval.jsonl",
        stratified,
        text_by_uid,
        eval_slice="confirmatory_coverage_stratified_stageA_secondary",
    )
    written_broad_uids = _load_uids(Path(broad_summary["path"]))
    written_stratified_uids = _load_uids(Path(stratified_summary["path"]))
    manifest = {
        "schema_version": "slm-confirmatory-holdouts-v1",
        "status": "frozen_before_confirmatory_training_outcomes",
        "dataset": dataset,
        "profile": profile,
        "created_date": "2026-06-10",
        "primary_eval": "confirmatory_broad_stageA_eval",
        "secondary_eval": "confirmatory_coverage_stratified_stageA_eval",
        "primary_success_scope": "Only the broad Stage-A holdout can determine the primary confirmatory result.",
        "secondary_scope": "Coverage-stratified holdout is a mechanism diagnostic and cannot rescue a failed primary result.",
        "utility_scope": "Stage C validation only; never selector objective",
        "selection_contract": {
            "primary": {
                "source": "Stage-A-passing records remaining after all frozen train arms and the legacy diagnostic holdout are excluded.",
                "sampling": "stable hash random sampling",
                "seed": int(broad_seed),
                "target_word_budget": int(broad_word_budget),
            },
            "secondary": {
                "source": "Remaining Stage-A-passing records after primary holdout exclusion.",
                "sampling": "round-robin stable hash sampling over style_bucket x length_bucket strata",
                "seed": int(stratified_seed),
                "target_word_budget": int(stratified_word_budget),
            },
        },
        "excluded_sources": excluded_sources,
        "candidate_stageA_records_after_exclusions": len(candidates),
        "holdouts": {
            "confirmatory_broad_stageA_eval": broad_summary,
            "confirmatory_coverage_stratified_stageA_eval": stratified_summary,
        },
        "disjointness": {
            "primary_vs_excluded_training_and_legacy_eval_overlap": len(written_broad_uids & excluded_uids),
            "secondary_vs_excluded_training_and_legacy_eval_overlap": len(written_stratified_uids & excluded_uids),
            "primary_vs_secondary_overlap": len(written_broad_uids & written_stratified_uids),
            "exact_uid_disjoint": not (
                (written_broad_uids & excluded_uids)
                or (written_stratified_uids & excluded_uids)
                or (written_broad_uids & written_stratified_uids)
            ),
        },
        "claim_boundary": "Internal same-corpus untouched outcome evaluation; external benchmarks and near-duplicate contamination audits remain required.",
    }
    save_json(experiment_dir / "confirmatory_holdouts_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare untouched confirmatory SLM holdouts.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--index-db", type=Path, default=DEFAULT_INDEX_DB_PATH)
    parser.add_argument("--broad-seed", type=int, default=20260612)
    parser.add_argument("--stratified-seed", type=int, default=20260613)
    parser.add_argument("--broad-word-budget", type=int, default=1000000)
    parser.add_argument("--stratified-word-budget", type=int, default=1000000)
    args = parser.parse_args()
    manifest = build_holdouts(
        experiment_dir=args.experiment_dir,
        profiles_path=args.profiles,
        index_db_path=args.index_db,
        broad_seed=int(args.broad_seed),
        stratified_seed=int(args.stratified_seed),
        broad_word_budget=int(args.broad_word_budget),
        stratified_word_budget=int(args.stratified_word_budget),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
