#!/usr/bin/env python3
"""Normalize raw candidate records and split release/quarantine outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, save_json
from ingestion.normalize import process_candidate


DEFAULT_INPUT = Path(__file__).resolve().parent / "validation" / "fixtures" / "stage0_raw_candidates.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "stage0"


def _load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Stage-0 raw input must contain a list of record objects.")
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_file(input_path: Path, output_dir: Path) -> Dict[str, Any]:
    processed = [process_candidate(record, index=index) for index, record in enumerate(_load_records(input_path))]
    release = [record for record in processed if record["release_eligibility"]["eligible"]]
    quarantine = [record for record in processed if not record["release_eligibility"]["eligible"]]
    reason_counts: Dict[str, int] = {}
    for record in quarantine:
        for reason in record["quarantine"]["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    _write_jsonl(output_dir / "release_candidates.jsonl", release)
    _write_jsonl(output_dir / "quarantined_candidates.jsonl", quarantine)
    report = {
        "schema_version": "stage0-processing-report-v1",
        "input": str(input_path),
        "contract": "candidate-corpus-record-v1",
        "summary": {
            "input_records": len(processed),
            "release_candidate_records": len(release),
            "quarantined_records": len(quarantine),
            "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        },
        "outputs": {
            "release_candidates": str(output_dir / "release_candidates.jsonl"),
            "quarantined_candidates": str(output_dir / "quarantined_candidates.jsonl"),
        },
        "records": {
            record["record_id"]: {
                "status": record["quarantine"]["status"],
                "reasons": record["quarantine"]["reasons"],
                "transformations": record["transformations"],
            }
            for record in processed
        },
    }
    save_json(output_dir / "stage0_processing_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Process Stage-0 raw candidate records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = process_file(args.input, args.output_dir)
    summary = report["summary"]
    print(
        f"[30] Stage-0 processing: input={summary['input_records']} "
        f"release={summary['release_candidate_records']} quarantine={summary['quarantined_records']}"
    )
    print(f"[30] report: {args.output_dir / 'stage0_processing_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
