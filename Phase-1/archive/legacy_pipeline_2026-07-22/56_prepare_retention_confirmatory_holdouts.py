#!/usr/bin/env python3
"""Prepare untouched target and external holdouts for retention confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import INDEX_DIR, OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file
from policy.subsets import SCORING_MANIFEST_PATH, _passes_gates, _stage_a_gate


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_WIKITEXT_BATCH_DIR = Path("validation") / "fixtures" / "wikitext103_subset"
DEFAULT_PROFILES = Path("configs") / "curation_profiles.json"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip()


def _text_hash(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _minhash_signature(text: str, width: int = 5, count: int = 8) -> Tuple[str, ...]:
    words = _normalize(text).split()
    if len(words) < width:
        return (_text_hash(text),)
    hashes = {
        hashlib.blake2b(" ".join(words[idx : idx + width]).encode("utf-8"), digest_size=8).hexdigest()
        for idx in range(len(words) - width + 1)
    }
    return tuple(sorted(hashes)[:count])


def _iter_json_batch_records(batch_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(batch_dir.glob("batch_*.json")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        for record in payload:
            if isinstance(record, dict):
                yield record


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    words = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            words += len(str(record.get("text") or "").split())
    return {"path": str(path), "sha256": sha256_file(path), "records": count, "word_count": words}


def _training_fingerprints(paths: Iterable[Path]) -> Tuple[set[str], set[Tuple[str, ...]]]:
    hashes: set[str] = set()
    signatures: set[Tuple[str, ...]] = set()
    for path in paths:
        for record in iter_jsonl_records_resilient(path):
            text = str(record.get("text") or "")
            if not text.strip():
                continue
            hashes.add(_text_hash(text))
            signatures.add(_minhash_signature(text))
    return hashes, signatures


def prepare_holdouts(
    experiment_dir: Path,
    wikitext_batch_dir: Path,
    profiles_path: Path,
    index_db: Path,
    target_word_budget: int,
    external_word_budget: int,
) -> Dict[str, Any]:
    helpers = importlib.import_module("44_prepare_slm_confirmatory_holdouts")
    experiment_manifest = load_json(experiment_dir / "manifest.json")
    dataset = str(experiment_manifest["dataset"])
    profile = str(experiment_manifest["profile"])
    stage_a = _stage_a_gate((load_json(profiles_path).get("profiles") or {})[profile])
    scored_path = Path(
        str(((load_json(SCORING_MANIFEST_PATH).get("datasets") or {}).get(dataset) or {})["path"])
    )
    train_paths = [
        experiment_dir / "retention_replay_target099.jsonl",
        experiment_dir / "stageA_random_equal_budget.jsonl",
    ]
    previous_eval_paths = [
        experiment_dir / "heldout_stageA_eval.jsonl",
        experiment_dir / "confirmatory_broad_stageA_eval.jsonl",
        experiment_dir / "confirmatory_coverage_stratified_stageA_eval.jsonl",
        experiment_dir / "external_guardrails" / "wikitext103_validation_test_guardrail.jsonl",
    ]
    excluded_uids: set[str] = set()
    for path in train_paths + previous_eval_paths[:3]:
        excluded_uids.update(helpers._load_uids(path))
    train_hashes, train_signatures = _training_fingerprints(train_paths)
    previous_external_hashes = {
        _text_hash(str(record.get("text") or ""))
        for record in iter_jsonl_records_resilient(previous_eval_paths[3])
    }

    target_candidates = []
    for record in iter_jsonl_records_resilient(scored_path):
        uid = helpers._uid(record)
        if not uid or uid in excluded_uids or not _passes_gates(record, stage_a):
            continue
        target_candidates.append(record)
    ordered_target = helpers._take_stratified(target_candidates, budget_words=target_word_budget * 2, seed=20260614)
    target_uids = {helpers._uid(record) for record in ordered_target}
    text_by_uid = helpers._load_index_texts(index_db, target_uids)
    target_records: List[Dict[str, Any]] = []
    target_words = 0
    target_signature_rejections = 0
    for record in ordered_target:
        text = text_by_uid.get(helpers._uid(record), "")
        if not text.strip():
            continue
        if _text_hash(text) in train_hashes or _minhash_signature(text) in train_signatures:
            target_signature_rejections += 1
            continue
        target_records.append({"id": helpers._uid(record), "text": text, "source": record.get("source")})
        target_words += len(text.split())
        if target_words >= target_word_budget:
            break

    external_candidates = []
    for record in _iter_json_batch_records(wikitext_batch_dir):
        if str(record.get("source_split") or "") != "train":
            continue
        text = str(record.get("text") or "")
        text_hash = _text_hash(text)
        signature = _minhash_signature(text)
        if not text.strip() or text_hash in train_hashes or text_hash in previous_external_hashes:
            continue
        if signature in train_signatures:
            continue
        stable = hashlib.sha256(f"retention-confirmatory-external:{record.get('id')}".encode("utf-8")).hexdigest()
        external_candidates.append((stable, record))
    external_candidates.sort(key=lambda item: item[0])
    external_records = []
    external_words = 0
    for _stable, record in external_candidates:
        text = str(record.get("text") or "")
        external_records.append(
            {
                "id": str(record.get("id") or _text_hash(text)),
                "text": text,
                "source": "wikitext103_train_unused_confirmatory",
                "source_split": "train",
            }
        )
        external_words += len(text.split())
        if external_words >= external_word_budget:
            break

    target_summary = _write_jsonl(experiment_dir / "retention_confirmatory_target_eval.jsonl", target_records)
    external_summary = _write_jsonl(experiment_dir / "retention_confirmatory_external_eval.jsonl", external_records)
    manifest = {
        "schema_version": "retention-confirmatory-holdouts-v1",
        "status": "frozen_before_retention_confirmatory_training",
        "target_holdout": target_summary,
        "external_holdout": external_summary,
        "excluded_paths": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in train_paths + previous_eval_paths
        ],
        "disjointness": {
            "exact_uid_target_vs_train_and_previous_target": 0,
            "exact_normalized_text_external_vs_train_and_previous_external": 0,
            "coarse_minhash_signature_overlap_vs_train": 0,
            "target_signature_rejections": target_signature_rejections,
        },
        "near_duplicate_audit_scope": (
            "Exact collision of normalized 5-word-shingle MinHash signatures. "
            "This is a coarse near-duplicate control, not a semantic-contamination proof."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Untouched internal target and provisional external-corpus retention holdouts. "
            "Task benchmarks and embedding-based contamination audits remain required."
        ),
    }
    save_json(experiment_dir / "retention_confirmatory_holdouts_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare retention confirmatory holdouts.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--wikitext-batch-dir", type=Path, default=DEFAULT_WIKITEXT_BATCH_DIR)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--index-db", type=Path, default=INDEX_DIR / "index.sqlite")
    parser.add_argument("--target-word-budget", type=int, default=1000000)
    parser.add_argument("--external-word-budget", type=int, default=1000000)
    args = parser.parse_args()
    manifest = prepare_holdouts(
        args.experiment_dir,
        args.wikitext_batch_dir,
        args.profiles,
        args.index_db,
        int(args.target_word_budget),
        int(args.external_word_budget),
    )
    print({"target": manifest["target_holdout"], "external": manifest["external_holdout"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
