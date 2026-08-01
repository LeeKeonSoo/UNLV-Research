#!/usr/bin/env python3
"""Prepare a frozen external holdout and exact-overlap contamination audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, save_json, sha256_file


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_BATCH_DIR = Path("validation") / "fixtures" / "wikitext103_subset"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "external_guardrails"


def _normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _iter_batch_records(batch_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(batch_dir.glob("batch_*.json")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list: {path}")
        for record in payload:
            if isinstance(record, dict):
                yield record


def _training_hashes(experiment_dir: Path) -> Dict[str, set[str]]:
    arm_paths = {
        "selected_only": experiment_dir / "curated_equal_budget.jsonl",
        "coverage_backfilled": experiment_dir / "coverage_backfilled_interleaved50_equal_budget.jsonl",
        "stageA_broad": experiment_dir / "stageA_random_equal_budget.jsonl",
    }
    results: Dict[str, set[str]] = {}
    for arm, path in arm_paths.items():
        if not path.exists():
            continue
        results[arm] = {
            _normalized_text_hash(str(record.get("text") or ""))
            for record in iter_jsonl_records_resilient(path)
            if str(record.get("text") or "").strip()
        }
    return results


def prepare_holdout(batch_dir: Path, experiment_dir: Path, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wikitext103_validation_test_guardrail.jsonl"
    training_hashes = _training_hashes(experiment_dir)
    overlap_counts = {arm: 0 for arm in training_hashes}
    split_counts: Dict[str, int] = {}
    record_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for record in _iter_batch_records(batch_dir):
            split = str(record.get("source_split") or "")
            text = str(record.get("text") or "")
            if split not in {"validation", "test"} or not text.strip():
                continue
            text_hash = _normalized_text_hash(text)
            for arm, hashes in training_hashes.items():
                if text_hash in hashes:
                    overlap_counts[arm] += 1
            row = {
                "id": str(record.get("id") or f"wikitext_guardrail_{record_count:06d}"),
                "text": text,
                "source_split": split,
                "source": "wikitext103_subset",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_counts[split] = split_counts.get(split, 0) + 1
            record_count += 1

    if record_count == 0:
        raise RuntimeError("No WikiText validation/test records found.")

    manifest = {
        "schema_version": "external-guardrail-holdout-v1",
        "holdout_name": "wikitext103_validation_test_guardrail",
        "purpose": "Provisional external general-language retention and forgetting guardrail.",
        "source_batch_dir": str(batch_dir.resolve()),
        "source_splits": ["validation", "test"],
        "record_count": record_count,
        "split_counts": split_counts,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "exact_normalized_text_overlap_counts": overlap_counts,
        "exact_overlap_pass": all(count == 0 for count in overlap_counts.values()),
        "claim_boundary": (
            "This holdout supports a provisional external-corpus NLL retention check only. "
            "It is not a task benchmark, near-duplicate audit, or proof of model-pretraining decontamination."
        ),
    }
    save_json(output_dir / "external_guardrail_holdout_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare external guardrail holdout.")
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = prepare_holdout(args.batch_dir, args.experiment_dir, args.output_dir)
    print(
        {
            "records": manifest["record_count"],
            "exact_overlap_pass": manifest["exact_overlap_pass"],
            "output": manifest["output_path"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
