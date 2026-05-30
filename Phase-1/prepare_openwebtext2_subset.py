#!/usr/bin/env python3
"""Prepare a manageable OpenWebText2 subset as a prepared input dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from data_eval_common import DEFAULT_TOKENIZER_NAME, estimate_token_count


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "validation" / "fixtures" / "openwebtext2_subset"
DEFAULT_LIMIT = 500000
DEFAULT_TARGET_GB = 2.0
DEFAULT_SHUFFLE_BUFFER = 50000
BATCH_SIZE = 1000


def _text_from_record(record: Dict[str, Any]) -> str:
    for key in ("text", "content", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _write_batch(output_dir: Path, batch_idx: int, rows: List[Dict[str, Any]]) -> int:
    batch_path = output_dir / f"batch_{batch_idx:03d}.json"
    payload = json.dumps(rows, ensure_ascii=False)
    batch_path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def prepare_subset(
    limit: int,
    seed: int,
    output_path: Path,
    target_gb: float,
    shuffle_buffer_size: int,
    target_tokens: int | None,
    tokenizer_name: str,
) -> Path:
    ds = load_dataset("Geralt-Targaryen/openwebtext2", split="train", streaming=True)
    # Keep the shuffle buffer bounded so larger sample limits do not explode memory.
    ds = ds.shuffle(buffer_size=max(shuffle_buffer_size, 1000), seed=seed)

    output_dir = output_path
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    seen = 0
    bytes_written = 0
    tokens_written = 0
    target_bytes = None if target_tokens is not None else int(target_gb * (1024 ** 3))
    batch_idx = 0

    for record in tqdm(ds, total=limit, desc="[prep] openwebtext2 subset", unit="doc"):
        text = _text_from_record(record)
        if len(text.strip()) < 200:
            continue
        row = (
            {
                "id": f"owt2_{seen:06d}",
                "text": text,
            }
        )
        row_bytes = len(json.dumps(row, ensure_ascii=False).encode("utf-8"))
        row_tokens = estimate_token_count(text, tokenizer_name=tokenizer_name)
        exceeds_bytes = target_bytes is not None and bytes_written + row_bytes > target_bytes and rows
        exceeds_tokens = target_tokens is not None and tokens_written + row_tokens > target_tokens and rows
        if exceeds_bytes or exceeds_tokens:
            break
        rows.append(row)
        seen += 1
        tokens_written += row_tokens
        if len(rows) >= BATCH_SIZE:
            bytes_written += _write_batch(output_dir, batch_idx, rows)
            batch_idx += 1
            rows = []
        hit_bytes_target = target_bytes is not None and bytes_written >= target_bytes
        hit_token_target = target_tokens is not None and tokens_written >= target_tokens
        if seen >= limit or hit_bytes_target or hit_token_target:
            break

    if rows:
        bytes_written += _write_batch(output_dir, batch_idx, rows)

    manifest = {
        "dataset": "openwebtext2_subset",
        "requested_limit": limit,
        "target_gb": target_gb,
        "target_tokens": target_tokens,
        "tokenizer_name": tokenizer_name,
        "shuffle_buffer_size": shuffle_buffer_size,
        "approx_bytes_written": bytes_written,
        "approx_tokens_written": tokens_written,
        "records_written": seen,
        "batch_count": batch_idx + (1 if rows else 0),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare an OpenWebText2 subset for contrast evaluation.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-gb", type=float, default=DEFAULT_TARGET_GB)
    parser.add_argument("--shuffle-buffer-size", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    args = parser.parse_args()
    path = prepare_subset(
        limit=args.limit,
        seed=args.seed,
        output_path=args.output,
        target_gb=args.target_gb,
        shuffle_buffer_size=args.shuffle_buffer_size,
        target_tokens=args.target_tokens,
        tokenizer_name=args.tokenizer_name,
    )
    print(f"[08] wrote subset: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
