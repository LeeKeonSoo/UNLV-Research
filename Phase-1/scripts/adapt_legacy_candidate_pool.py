#!/usr/bin/env python3
"""Translate collector-era JSONL rows into the current source-preserving input contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _mapping(value: Any) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _legacy_partition(row: JsonMap, source: JsonMap) -> JsonMap:
    """Preserve legacy collection fields as audit metadata without domain authority."""
    defaults = _mapping(source.get("partition"))
    return {
        **defaults,
        "source_dataset": str(row.get("source_dataset_id") or source["source_name"]),
        "source_split": row.get("source_split"),
        "source_row_index": row.get("source_row_index"),
        "legacy_domain_label": row.get("domain"),
        "legacy_pool_role": row.get("pool_role"),
        "provenance_reconstruction": "legacy_collection_manifest_and_row_fields",
    }


def adapt_rows(rows: list[JsonMap], source: JsonMap) -> list[JsonMap]:
    """Return canonical candidates while retaining unresolved upstream rights."""
    rights = _mapping(source.get("rights"))
    language = _mapping(source.get("language"))
    return [
        {
            "record_id": str(row.get("record_uid") or row.get("record_id") or f"legacy-{index:08d}"),
            "text": str(row.get("text") or ""),
            "provenance": {
                "source_name": str(source["source_name"]),
                "source_uri": str(source["source_uri"]),
                "collected_at": str(source["collected_at"]),
            },
            "language": {"code": str(language.get("code") or "und"), "confidence": language.get("confidence")},
            "rights": {"status": str(rights.get("status") or "unknown"), "license": rights.get("license")},
            "pii_context": str(source.get("pii_context") or "general"),
            "partition": _legacy_partition(row, source),
        }
        for index, row in enumerate(rows)
    ]


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [row for line in handle if isinstance((row := json.loads(line)), dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Adapt legacy collector JSONL into current curation input JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _mapping(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    required = ("source_name", "source_uri", "collected_at")
    missing = [field for field in required if not isinstance(source.get(field), str) or not str(source[field]).strip()]
    if missing:
        raise RuntimeError(f"Source manifest missing required fields: {', '.join(missing)}")
    rows = adapt_rows(_read_jsonl(args.input), source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(rows), "rights_status": rows[0]["rights"]["status"] if rows else "unknown", "source_name": source["source_name"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
