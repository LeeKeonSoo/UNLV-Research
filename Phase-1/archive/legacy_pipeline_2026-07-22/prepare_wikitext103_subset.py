#!/usr/bin/env python3
"""Prepare a WikiText-103 raw subset with a token budget comparable to existing inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm import tqdm

from data_eval_common import DEFAULT_TOKENIZER_NAME, estimate_token_count


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "validation" / "fixtures" / "wikitext103_subset"
DEFAULT_TARGET_TOKENS = 250_000_000
DEFAULT_MIN_DOC_CHARS = 1_000
BATCH_SIZE = 1000


HEADING_RE = re.compile(r"^\s*=\s+(.+?)\s+=\s*$")


def _clean_line(text: Any) -> str:
    line = str(text or "").strip()
    line = re.sub(r"\s+", " ", line)
    return line


def _heading_title(line: str) -> str | None:
    match = HEADING_RE.match(line)
    if not match:
        return None
    title = re.sub(r"\s+", " ", match.group(1)).strip(" =")
    return title or None


def _iter_wikitext_docs(split: str) -> Iterable[Dict[str, str]]:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    title = "untitled"
    parts: List[str] = []
    article_idx = 0

    def flush() -> Dict[str, str] | None:
        nonlocal article_idx, parts, title
        body = "\n\n".join(part for part in parts if part.strip()).strip()
        if not body:
            parts = []
            return None
        article_idx += 1
        payload = {
            "id": f"wikitext103_{split}_{article_idx:07d}",
            "title": title,
            "text": body,
            "source_split": split,
            "license": "CC BY-SA 4.0",
        }
        parts = []
        return payload

    for row in dataset:
        line = _clean_line(row.get("text"))
        if not line:
            continue
        heading = _heading_title(line)
        if heading is not None:
            doc = flush()
            if doc is not None:
                yield doc
            title = heading
            parts = [heading]
            continue
        parts.append(line)

    doc = flush()
    if doc is not None:
        yield doc


def _write_batch(output_dir: Path, batch_idx: int, rows: List[Dict[str, str]]) -> int:
    path = output_dir / f"batch_{batch_idx:03d}.json"
    payload = json.dumps(rows, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    return len(payload.encode("utf-8"))


def prepare_wikitext103_subset(
    *,
    output_path: Path,
    target_tokens: int,
    tokenizer_name: str,
    min_doc_chars: int,
    splits: List[str],
) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    batch_idx = 0
    docs_written = 0
    docs_seen = 0
    bytes_written = 0
    tokens_written = 0
    target = max(1, int(target_tokens))

    for split in splits:
        for doc in tqdm(_iter_wikitext_docs(split), desc=f"[prep] wikitext103 {split}", unit="doc"):
            docs_seen += 1
            text = str(doc.get("text") or "").strip()
            if len(text) < int(min_doc_chars):
                continue
            doc_tokens = estimate_token_count(text, tokenizer_name=tokenizer_name)
            if rows and tokens_written + doc_tokens > target:
                break
            rows.append(doc)
            docs_written += 1
            tokens_written += int(doc_tokens)
            if len(rows) >= BATCH_SIZE:
                bytes_written += _write_batch(output_path, batch_idx, rows)
                batch_idx += 1
                rows = []
            if tokens_written >= target:
                break
        if tokens_written >= target:
            break

    if rows:
        bytes_written += _write_batch(output_path, batch_idx, rows)
        batch_idx += 1

    manifest = {
        "dataset": "wikitext103_subset",
        "hf_dataset": "Salesforce/wikitext",
        "hf_config": "wikitext-103-raw-v1",
        "splits": splits,
        "target_tokens": int(target_tokens),
        "tokenizer_name": tokenizer_name,
        "min_doc_chars": int(min_doc_chars),
        "approx_tokens_written": int(tokens_written),
        "approx_bytes_written": int(bytes_written),
        "docs_seen": int(docs_seen),
        "records_written": int(docs_written),
        "batch_count": int(batch_idx),
        "license": "CC BY-SA 4.0",
        "note": "If approx_tokens_written is below target_tokens, the full available WikiText-103 raw split budget was exhausted.",
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare WikiText-103 raw as a Phase-1 input dataset.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--min-doc-chars", type=int, default=DEFAULT_MIN_DOC_CHARS)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()
    path = prepare_wikitext103_subset(
        output_path=args.output,
        target_tokens=args.target_tokens,
        tokenizer_name=args.tokenizer_name,
        min_doc_chars=args.min_doc_chars,
        splits=[str(x) for x in args.splits],
    )
    print(f"[prep] wrote WikiText-103 subset: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
