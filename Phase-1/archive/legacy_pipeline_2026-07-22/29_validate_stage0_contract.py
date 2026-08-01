#!/usr/bin/env python3
"""Validate Stage-0 candidate records and release/quarantine decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, save_json
from ingestion.schema import CANDIDATE_RECORD_SCHEMA_VERSION, release_eligibility


DEFAULT_INPUT = Path(__file__).resolve().parent / "validation" / "fixtures" / "stage0_candidate_records.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage0_contract_validation.json"


def _records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Stage-0 input must be a list or an object with a records list.")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every Stage-0 candidate record must be an object.")
        yield row


def build_report(path: Path) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for index, record in enumerate(_records(path)):
        record_id = str(record.get("record_id") or f"record_{index}")
        records[record_id] = {
            "declared_status": (record.get("quarantine") or {}).get("status"),
            **release_eligibility(record),
        }
    invalid = [record_id for record_id, result in records.items() if result["validation_errors"]]
    eligible = [record_id for record_id, result in records.items() if result["eligible"]]
    return {
        "schema_version": "stage0-contract-validation-v1",
        "candidate_record_schema_version": CANDIDATE_RECORD_SCHEMA_VERSION,
        "input": str(path),
        "summary": {
            "record_count": len(records),
            "contract_valid_count": len(records) - len(invalid),
            "release_eligible_count": len(eligible),
            "quarantined_or_rejected_count": len(records) - len(eligible),
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage-0 candidate-corpus contract.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.input)
    save_json(args.output, report)
    summary = report["summary"]
    print(
        f"[29] Stage-0 contract: records={summary['record_count']} "
        f"valid={summary['contract_valid_count']} release_eligible={summary['release_eligible_count']}"
    )
    print(f"[29] report: {args.output}")
    return 0 if summary["contract_valid_count"] == summary["record_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
