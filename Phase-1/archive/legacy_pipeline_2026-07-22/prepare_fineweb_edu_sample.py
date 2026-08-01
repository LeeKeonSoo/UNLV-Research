#!/usr/bin/env python3
"""Prepare a manageable FineWeb-Edu sample as a Phase-1 input dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from data_eval_common import DEFAULT_TOKENIZER_NAME, estimate_token_count


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "validation" / "fixtures" / "fineweb_edu_sample"
DEFAULT_HF_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_HF_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_LIMIT = 500000
DEFAULT_TARGET_TOKENS = 250_000_000
DEFAULT_MIN_TEXT_CHARS = 300
DEFAULT_SHUFFLE_BUFFER = 50000
BATCH_SIZE = 1000


def _text_from_record(record: Dict[str, Any]) -> str:
    for key in ("text", "content", "document"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _metadata_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in (
        "id",
        "url",
        "dump",
        "file_path",
        "language",
        "language_score",
        "score",
        "int_score",
        "token_count",
    ):
        value = record.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _stable_id(prefix: str, seen: int, text: str, metadata: Dict[str, Any]) -> str:
    raw_id = metadata.get("id")
    if raw_id is not None and str(raw_id).strip():
        return f"{prefix}_{str(raw_id).strip()}"
    digest = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{prefix}_{seen:08d}_{digest}"


def _write_batch(output_dir: Path, batch_idx: int, rows: List[Dict[str, Any]]) -> int:
    batch_path = output_dir / f"batch_{batch_idx:03d}.json"
    payload = json.dumps(rows, ensure_ascii=False)
    batch_path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def prepare_fineweb_edu_sample(
    *,
    output_path: Path,
    hf_dataset: str,
    hf_config: str,
    split: str,
    limit: int,
    seed: int,
    target_tokens: int,
    tokenizer_name: str,
    min_text_chars: int,
    shuffle_buffer_size: int,
) -> Path:
    ds = load_dataset(hf_dataset, name=hf_config, split=split, streaming=True)
    ds = ds.shuffle(buffer_size=max(int(shuffle_buffer_size), 1000), seed=int(seed))

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    records_seen = 0
    records_written = 0
    bytes_written = 0
    tokens_written = 0
    batch_idx = 0
    target = max(1, int(target_tokens))

    for record in tqdm(ds, total=int(limit), desc="[prep] fineweb-edu sample", unit="doc"):
        records_seen += 1
        text = _text_from_record(record)
        if len(text.strip()) < int(min_text_chars):
            if records_seen >= int(limit):
                break
            continue
        token_count = estimate_token_count(text, tokenizer_name=tokenizer_name)
        if rows and tokens_written + token_count > target:
            break
        metadata = _metadata_from_record(record)
        row = {
            "id": _stable_id("fineweb_edu", records_written, text, metadata),
            "text": text,
            "source_dataset": hf_dataset,
            "source_config": hf_config,
            "source_split": split,
            "license": "ODC-By-1.0",
        }
        if metadata:
            row["source_metadata"] = metadata
        rows.append(row)
        records_written += 1
        tokens_written += int(token_count)
        if len(rows) >= BATCH_SIZE:
            bytes_written += _write_batch(output_path, batch_idx, rows)
            batch_idx += 1
            rows = []
        if records_seen >= int(limit) or tokens_written >= target:
            break

    if rows:
        bytes_written += _write_batch(output_path, batch_idx, rows)
        batch_idx += 1

    manifest = {
        "dataset": "fineweb_edu_sample",
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "split": split,
        "requested_limit": int(limit),
        "target_tokens": int(target_tokens),
        "tokenizer_name": tokenizer_name,
        "min_text_chars": int(min_text_chars),
        "shuffle_buffer_size": int(shuffle_buffer_size),
        "seed": int(seed),
        "approx_tokens_written": int(tokens_written),
        "approx_bytes_written": int(bytes_written),
        "records_seen": int(records_seen),
        "records_written": int(records_written),
        "batch_count": int(batch_idx),
        "license": "ODC-By-1.0",
        "common_crawl_terms_apply": True,
        "role": "clean_demonstration_dataset_for_lm_training_curation",
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a FineWeb-Edu sample for Phase-1 curation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-config", default=DEFAULT_HF_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--min-text-chars", type=int, default=DEFAULT_MIN_TEXT_CHARS)
    parser.add_argument("--shuffle-buffer-size", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    args = parser.parse_args()
    path = prepare_fineweb_edu_sample(
        output_path=args.output,
        hf_dataset=str(args.hf_dataset),
        hf_config=str(args.hf_config),
        split=str(args.split),
        limit=int(args.limit),
        seed=int(args.seed),
        target_tokens=int(args.target_tokens),
        tokenizer_name=str(args.tokenizer_name),
        min_text_chars=int(args.min_text_chars),
        shuffle_buffer_size=int(args.shuffle_buffer_size),
    )
    print(f"[prep] wrote FineWeb-Edu sample: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
