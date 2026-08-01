#!/usr/bin/env python3
"""Merge base and expansion Stage-0 pools before rerunning Stage A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json, sha256_file


DEFAULT_INPUT_DIRS = [
    OUTPUT_DIR / "temporal_code_collection" / "stage0_path_stratified_tranche",
    OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_expansion",
]
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined"
SPLITS = ("train", "development", "confirmatory")
FILES = ("release_candidates.jsonl", "quarantined_candidates.jsonl")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge(input_dirs: List[Path], output_dir: Path) -> Dict[str, Any]:
    source_sha256 = {
        str(path / split / file_name): sha256_file(path / split / file_name)
        for path in input_dirs
        for split in SPLITS
        for file_name in FILES
        if (path / split / file_name).exists()
    }
    split_counts = {}
    duplicate_record_ids = []
    for split in SPLITS:
        split_counts[split] = {}
        for file_name in FILES:
            rows = []
            seen = set()
            for input_dir in input_dirs:
                source_label = input_dir.name
                for row in _jsonl(input_dir / split / file_name):
                    record_id = str(row.get("record_id") or row.get("id") or "")
                    if record_id in seen:
                        duplicate_record_ids.append(record_id)
                    seen.add(record_id)
                    tagged = dict(row)
                    tagged["code_domain_v2_source_pool"] = source_label
                    rows.append(tagged)
            _write_jsonl(output_dir / split / file_name, rows)
            split_counts[split][file_name.removesuffix(".jsonl")] = len(rows)
    report = {
        "schema_version": "code-domain-v2-stage0-combined-pool-v1",
        "status": "stage0_pools_merged_before_stage_a",
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "source_sha256": source_sha256,
        "summary": {
            "split_counts": split_counts,
            "duplicate_record_id_count": len(duplicate_record_ids),
            "duplicate_record_id_examples": duplicate_record_ids[:20],
        },
        "next_step": "Run 74_run_temporal_code_stage_a_smoke.py on this combined Stage-0 pool.",
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Stage-0 merge only; no Stage-A, Stage-B, Stage-C, Utility, or training claim.",
    }
    save_json(output_dir / "stage0_combined_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge code-domain v2 Stage-0 pools.")
    parser.add_argument("--input-dir", type=Path, action="append", dest="input_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = merge(args.input_dirs or DEFAULT_INPUT_DIRS, args.output_dir)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
