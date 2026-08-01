#!/usr/bin/env python3
"""Materialize repository-disjoint The Stack V2 source samples via Software Heritage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]
SOURCE_URI = "https://huggingface.co/datasets/bigcode/the-stack-v2"


def _sample_index(repository: str, assignment_seed: str, sample_count: int) -> int:
    digest = hashlib.sha256(f"{assignment_seed}::{repository}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % sample_count


def _artifact_context(row: JsonMap) -> JsonMap:
    return {
        "generation": "generated" if row.get("is_generated") is True else "authored",
        "dependency_copy": row.get("is_vendor") is True,
    }


def _record(row: JsonMap, text: str, sample_id: int, token_count: int) -> JsonMap:
    blob_id = str(row["blob_id"])
    content_id = str(row.get("content_id") or blob_id)
    repository = str(row["repo_name"])
    return {
        "record_id": f"the-stack-v2::{content_id}",
        "text": text,
        "token_count": token_count,
        "provenance": {"source_name": "bigcode/the-stack-v2", "source_uri": SOURCE_URI, "collected_at": datetime.now(UTC).isoformat()},
        "language": {"code": "python", "confidence": 1.0},
        "artifact_context": _artifact_context(row),
        "rights": {"status": "allowed", "license": "permissive"},
        "pii_context": "repository_code",
        "partition": {
            "source_dataset": "bigcode/the-stack-v2",
            "source_pool_role": "raw_like",
            "content_type": "code",
            "repository_identity": repository,
            "path": str(row.get("path") or ""),
            "sample_id": sample_id + 1,
            "source_blob_id": blob_id,
            "source_content_id": content_id,
            "source_encoding": str(row.get("src_encoding") or ""),
        },
    }


def collect_repository_disjoint_samples(
    upstream: Iterable[JsonMap],
    *,
    fetch_content: Callable[[JsonMap], str],
    count_tokens: Callable[[str], int],
    sample_count: int,
    target_tokens: int,
    assignment_seed: str,
) -> tuple[list[list[JsonMap]], JsonMap]:
    """Collect whole permissively licensed files into stable repository-disjoint samples."""
    if sample_count < 1 or target_tokens < 1:
        raise RuntimeError("sample_count and target_tokens must be positive.")
    samples: list[list[JsonMap]] = [[] for _ in range(sample_count)]
    tokens = [0 for _ in range(sample_count)]
    repositories = [set() for _ in range(sample_count)]
    report: JsonMap = {"scanned_rows": 0, "skipped_by_license": 0, "skipped_full_sample": 0, "skipped_empty_content": 0, "fetch_failures": 0}
    for row in upstream:
        report["scanned_rows"] += 1
        if row.get("license_type") != "permissive":
            report["skipped_by_license"] += 1
            continue
        repository = row.get("repo_name")
        blob_id = row.get("blob_id")
        if not isinstance(repository, str) or not repository.strip() or not isinstance(blob_id, str) or not blob_id.strip():
            report["fetch_failures"] += 1
            continue
        index = _sample_index(repository, assignment_seed, sample_count)
        if tokens[index] >= target_tokens:
            report["skipped_full_sample"] += 1
            continue
        try:
            text = fetch_content(row)
        except (OSError, UnicodeError):
            report["fetch_failures"] += 1
            continue
        if not text.strip():
            report["skipped_empty_content"] += 1
            continue
        token_count = count_tokens(text)
        if token_count < 1:
            report["skipped_empty_content"] += 1
            continue
        samples[index].append(_record(row, text, index, token_count))
        tokens[index] += token_count
        repositories[index].add(repository)
        if all(total >= target_tokens for total in tokens):
            break
    report.update(
        {
            "sample_count": sample_count,
            "target_tokens_per_sample": target_tokens,
            "token_counts": tokens,
            "record_counts": [len(sample) for sample in samples],
            "repository_counts": [len(value) for value in repositories],
            "repository_disjoint": len(set().union(*repositories)) == sum(len(value) for value in repositories),
            "assignment_seed": assignment_seed,
            "admission": {"license_type": "permissive", "content_filters": "none"},
        }
    )
    return samples, report


def _load_token_counter(tokenizer_path: Path) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _load_s3_fetcher() -> Callable[[JsonMap], str]:
    import boto3
    from smart_open import open as smart_open

    client = boto3.Session().client("s3")

    def fetch(row: JsonMap) -> str:
        with smart_open(
            f"s3://softwareheritage/content/{row['blob_id']}",
            "rb",
            compression=".gz",
            transport_params={"client": client},
        ) as handle:
            return handle.read().decode(str(row["src_encoding"]))

    return fetch


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect repository-disjoint The Stack V2 Python samples.")
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--assignment-seed", default="the-stack-v2-python-raw-like-v1")
    parser.add_argument("--shuffle-seed", type=int, default=20260731)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    args = parser.parse_args()
    from dotenv import load_dotenv
    from datasets import load_dataset

    if not load_dotenv(args.dotenv):
        raise RuntimeError(f"Could not load dotenv file: {args.dotenv}")
    upstream = load_dataset("bigcode/the-stack-v2", "Python", split="train", streaming=True).shuffle(
        seed=args.shuffle_seed,
        buffer_size=args.shuffle_buffer,
    )
    samples, report = collect_repository_disjoint_samples(
        upstream,
        fetch_content=_load_s3_fetcher(),
        count_tokens=_load_token_counter(args.tokenizer_path),
        sample_count=args.sample_count,
        target_tokens=args.target_tokens,
        assignment_seed=args.assignment_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples, start=1):
        _write_jsonl(args.output_dir / f"raw_sample_{index:02d}.jsonl", sample)
    (args.output_dir / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
