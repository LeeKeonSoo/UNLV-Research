#!/usr/bin/env python3
"""Prepare the gated TinyTextbooks dataset as Phase-1 JSON batches."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "tiny_textbooks_raw"
DEFAULT_DATASET = "nampdn-ai/tiny-textbooks"
DEFAULT_SPLITS = ["train"]
DEFAULT_TEXT_FIELD = "textbook"
DEFAULT_BATCH_SIZE = 1000
DEFAULT_MIN_TEXT_CHARS = 100


def _text_from_record(record: Dict[str, Any], text_field: str) -> str:
    value = record.get(text_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    for key in ("textbook", "text", "content", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _write_batch(output_dir: Path, batch_idx: int, rows: List[Dict[str, Any]]) -> int:
    path = output_dir / f"batch_{batch_idx:03d}.json"
    payload = json.dumps(rows, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def prepare_tiny_textbooks(
    *,
    dataset_name: str,
    splits: List[str],
    output_path: Path,
    text_field: str,
    min_text_chars: int,
    batch_size: int,
    limit: int | None,
) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    batch_idx = 0
    records_seen = 0
    records_written = 0
    bytes_written = 0
    per_split_written: Dict[str, int] = {}

    for split in splits:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        per_split_written[split] = 0
        for record in tqdm(ds, desc=f"[prep] tiny_textbooks {split}", unit="doc"):
            records_seen += 1
            text = _text_from_record(record, text_field=text_field)
            if len(text) < int(min_text_chars):
                continue
            source_idx = record.get("idx")
            row = {
                "id": f"tiny_textbooks_{split}_{records_written:07d}",
                "text": text,
                "source_split": split,
                "source_idx": str(source_idx) if source_idx is not None else "",
            }
            source = record.get("source")
            if source is not None:
                row["source"] = str(source)
            rows.append(row)
            records_written += 1
            per_split_written[split] += 1
            if len(rows) >= int(batch_size):
                bytes_written += _write_batch(output_path, batch_idx, rows)
                batch_idx += 1
                rows = []
            if limit is not None and records_written >= int(limit):
                break
        if limit is not None and records_written >= int(limit):
            break

    if rows:
        bytes_written += _write_batch(output_path, batch_idx, rows)
        batch_idx += 1

    manifest = {
        "dataset": "tiny_textbooks",
        "source_hf_dataset": dataset_name,
        "splits": splits,
        "text_field_mapped_to_text": text_field,
        "min_text_chars": int(min_text_chars),
        "limit": limit,
        "records_seen": int(records_seen),
        "records_written": int(records_written),
        "records_written_by_split": per_split_written,
        "approx_bytes_written": int(bytes_written),
        "batch_size": int(batch_size),
        "batch_count": int(batch_idx),
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TinyTextbooks as Phase-1 input JSON batches.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    path = prepare_tiny_textbooks(
        dataset_name=str(args.dataset_name),
        splits=[str(split) for split in args.splits],
        output_path=args.output,
        text_field=str(args.text_field),
        min_text_chars=int(args.min_text_chars),
        batch_size=int(args.batch_size),
        limit=args.limit,
    )
    print(f"[prep] wrote TinyTextbooks dataset: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
