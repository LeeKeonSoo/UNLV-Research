#!/usr/bin/env python3
"""Freeze a source-declared reference pool without scoring candidate records."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _source_tier(row: JsonMap) -> str:
    partition = row.get("partition")
    if not isinstance(partition, dict):
        return "unlabeled"
    tier = partition.get("source_tier")
    return str(tier) if isinstance(tier, str) and tier else "unlabeled"


def _token_proxy(row: JsonMap) -> int:
    return len(str(row.get("text") or "").split())


def build_reference_pool(rows: Iterable[JsonMap], reference_source_tier: str) -> tuple[list[JsonMap], JsonMap]:
    """Select only source-declared reference records and describe the exclusion boundary."""
    all_rows = list(rows)
    selected = [row for row in all_rows if _source_tier(row) == reference_source_tier]
    source_datasets: Counter[str] = Counter()
    for row in selected:
        partition = row.get("partition")
        source_datasets[str(partition.get("source_dataset") or "unlabeled")] += 1 if isinstance(partition, dict) else 1
    return selected, {
        "schema_version": "calibrated-selector-reference-pool-v1",
        "status": "reference_pool_frozen_pending_hash_materialization",
        "selection_basis": "source_declared_source_tier_only",
        "reference_source_tier": reference_source_tier,
        "summary": {
            "input_records": len(all_rows),
            "reference_records": len(selected),
            "reference_whitespace_token_proxy": sum(_token_proxy(row) for row in selected),
            "selector_candidate_records_excluding_reference": len(all_rows) - len(selected),
        },
        "reference_source_datasets": dict(sorted(source_datasets.items())),
        "selector_boundary": {
            "reference_records_may_be_scored": False,
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
        },
        "claim_boundary": "Source-declared reference status is evidence for selector calibration only. It is not an intrinsic-quality label and does not authorize candidate removal.",
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a source-declared calibrated-selector reference pool.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-source-tier", required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    selected, report = build_reference_pool(_read_jsonl(args.input), args.reference_source_tier)
    args.reference_output.parent.mkdir(parents=True, exist_ok=True)
    with args.reference_output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report["input"] = {"path": str(args.input), "sha256": _sha256(args.input)}
    report["reference_output"] = {"path": str(args.reference_output), "sha256": _sha256(args.reference_output)}
    report["status"] = "reference_pool_frozen"
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
