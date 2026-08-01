#!/usr/bin/env python3
"""Adapt JSON batch arrays to the domain-neutral candidate input contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _mapping(value: Any) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _rows(directory: Path) -> list[JsonMap]:
    rows: list[JsonMap] = []
    for path in sorted(directory.glob("batch_*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(batch, list):
            raise RuntimeError(f"Batch is not a JSON array: {path}")
        rows.extend(row for row in batch if isinstance(row, dict))
    return rows


def _candidate(row: JsonMap, *, collected_at: str) -> JsonMap:
    source_metadata = _mapping(row.get("source_metadata"))
    source_name = str(row.get("source_dataset") or "undeclared-json-batch-source")
    source_uri = str(source_metadata.get("url") or "https://example.invalid/missing-source-uri")
    language = str(source_metadata.get("language") or "und")
    confidence = source_metadata.get("language_score")
    return {
        "record_id": str(row.get("id") or source_metadata.get("id") or "undeclared-record-id"),
        "text": str(row.get("text") or ""),
        "provenance": {"source_name": source_name, "source_uri": source_uri, "collected_at": collected_at},
        "language": {"code": language, "confidence": confidence if isinstance(confidence, (int, float)) else None},
        "rights": {"status": "allowed", "license": row.get("license")},
        "pii_context": "general",
        "partition": {
            "source_dataset": source_name,
            "source_tier": "raw_like",
            "content_type": "general_text",
            "source_config": row.get("source_config"),
            "source_split": row.get("source_split"),
        },
    }


def collect_rows(directory: Path, *, collected_at: str, limit: int) -> list[JsonMap]:
    """Return at most ``limit`` source-preserving candidate records from batches."""
    if limit <= 0:
        raise RuntimeError("Collection limit must be positive.")
    return [_candidate(row, collected_at=collected_at) for row in _rows(directory)[:limit]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect JSON batch arrays into a domain-neutral candidate JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--collected-at", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_rows(args.input_dir, collected_at=args.collected_at, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "source_datasets": sorted({str(row["partition"]["source_dataset"]) for row in rows})}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
