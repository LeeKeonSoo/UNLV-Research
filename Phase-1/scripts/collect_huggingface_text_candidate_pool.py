#!/usr/bin/env python3
"""Collect a bounded, source-preserving text slice from a Hugging Face dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class CollectionContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def _mapping(value: Any) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _token_proxy(text: str) -> int:
    return len(text.split())


def _text_field(source: JsonMap) -> str:
    value = source.get("text_field")
    return value if isinstance(value, str) and value.strip() else "text"


def _stable_record_id(row: JsonMap, source: JsonMap, index: int, digest: str) -> str:
    field = source.get("stable_record_id_field")
    value = row.get(field) if isinstance(field, str) and field.strip() else None
    stable_value = str(value).strip() if value is not None else ""
    suffix = stable_value if stable_value else f"{index:08d}::{digest}"
    return f"{source['source_name']}::{suffix}"


def _row_license(row: JsonMap, source: JsonMap) -> str | None:
    field = source.get("source_license_field")
    if not isinstance(field, str) or not field.strip():
        return None
    value = row.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        licenses = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return licenses[0] if licenses else None
    return None


def _allowed_license(row: JsonMap, source: JsonMap) -> str | None:
    allowed = source.get("allowed_source_licenses")
    if not isinstance(allowed, list):
        return _row_license(row, source) or str(_mapping(source.get("rights")).get("license") or "")
    allowed_licenses = {item for item in allowed if isinstance(item, str) and item.strip()}
    field = source.get("source_license_field")
    value = row.get(field) if isinstance(field, str) else None
    candidates = [value] if isinstance(value, str) else value if isinstance(value, list) else []
    for license_name in candidates:
        if isinstance(license_name, str) and license_name in allowed_licenses:
            return license_name
    return None


def _collection_admitted(row: JsonMap, source: JsonMap) -> bool:
    admission = _mapping(source.get("collection_admission"))
    text = str(row.get(_text_field(source)) or "")
    byte_count = len(text.encode("utf-8"))
    minimum_bytes = admission.get("minimum_bytes")
    maximum_bytes = admission.get("maximum_bytes")
    if isinstance(minimum_bytes, int) and byte_count < minimum_bytes:
        return False
    if isinstance(maximum_bytes, int) and byte_count > maximum_bytes:
        return False
    path_field = source.get("path_field")
    path = row.get(path_field) if isinstance(path_field, str) else None
    path_lower = path.lower() if isinstance(path, str) else ""
    excluded_fragments = admission.get("excluded_path_fragments")
    if isinstance(excluded_fragments, list) and any(
        isinstance(fragment, str) and fragment.lower() in path_lower for fragment in excluded_fragments
    ):
        return False
    return True


def _partition(source: JsonMap, row: JsonMap) -> JsonMap:
    partition = _mapping(source.get("partition"))
    repository_field = source.get("repository_identity_field")
    path_field = source.get("path_field")
    repository = row.get(repository_field) if isinstance(repository_field, str) else None
    path = row.get(path_field) if isinstance(path_field, str) else None
    return {
        **partition,
        "repository_identity": repository if isinstance(repository, str) else partition.get("repository_identity"),
        "path": path if isinstance(path, str) else partition.get("path"),
        "source_document_uri": row.get("url"),
        "source_document_date": row.get("date"),
        "source_row_metadata": row.get("metadata"),
    }


def _language(source: JsonMap, row: JsonMap) -> JsonMap:
    field = source.get("language_field")
    declared = row.get(field) if isinstance(field, str) and field.strip() else None
    if isinstance(declared, str) and declared.strip():
        return {
            "code": declared.strip().casefold(),
            "confidence": 1.0,
            "declaration": "source_row",
        }
    fallback = _mapping(source.get("language"))
    return {
        "code": str(fallback.get("code") or "und"),
        "confidence": fallback.get("confidence"),
    }


def _record(
    row: JsonMap,
    source: JsonMap,
    index: int,
    license_name: str,
    token_counter: TokenCounter,
) -> JsonMap:
    text = str(row.get(_text_field(source)) or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    token_count = token_counter(text)
    record: JsonMap = {
        "record_id": _stable_record_id(row, source, index, digest),
        "text": text,
        "token_proxy": token_count,
        "token_count": token_count,
        "provenance": {"source_name": str(source["source_name"]), "source_uri": str(source["source_uri"]), "collected_at": str(source["collected_at"])},
        "language": _language(source, row),
        "rights": {"status": str(_mapping(source.get("rights")).get("status") or "unknown"), "license": license_name},
        "pii_context": str(source.get("pii_context") or "general"),
        "partition": _partition(source, row),
    }
    record_shape = source.get("record_shape")
    if isinstance(record_shape, str) and record_shape.strip():
        record["record_shape"] = record_shape.strip()
    return record


def collect_rows(
    upstream: Iterable[JsonMap],
    source: JsonMap,
    *,
    token_limit: int,
    excluded_record_ids: set[str] | None = None,
    token_counter: TokenCounter = _token_proxy,
) -> list[JsonMap]:
    """Collect whole source rows through the first row reaching ``token_limit``."""
    if token_limit <= 0:
        raise CollectionContractError("Token limit must be positive.")
    rows: list[JsonMap] = []
    collected_tokens = 0
    excluded_ids = excluded_record_ids or set()
    for index, upstream_row in enumerate(upstream):
        license_name = _allowed_license(upstream_row, source)
        if license_name is None:
            continue
        if not _collection_admitted(upstream_row, source):
            continue
        row = _record(upstream_row, source, index, license_name, token_counter)
        if not row["text"].strip():
            continue
        if row["record_id"] in excluded_ids:
            continue
        rows.append(row)
        collected_tokens += int(row["token_proxy"])
        if collected_tokens >= token_limit:
            break
    return rows


def _excluded_record_ids(paths: Iterable[Path]) -> set[str]:
    record_ids: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record_id = json.loads(line).get("record_id")
                if isinstance(record_id, str) and record_id.strip():
                    record_ids.add(record_id)
    return record_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect a source-preserving Hugging Face text candidate pool.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--data-file", action="append", default=[])
    parser.add_argument("--revision")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--token-limit", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--exclude-record-id-jsonl", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _mapping(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    required = ("source_name", "source_uri", "collected_at")
    missing = [field for field in required if not isinstance(source.get(field), str) or not str(source[field]).strip()]
    if missing:
        raise CollectionContractError(f"Source manifest missing required fields: {', '.join(missing)}")
    from datasets import load_dataset

    load_options: JsonMap = {
        "revision": args.revision,
        "split": args.split,
        "streaming": True,
    }
    if args.data_file:
        load_options["data_files"] = args.data_file
    upstream = load_dataset(args.dataset, args.config, **load_options).shuffle(
        seed=args.shuffle_seed,
        buffer_size=args.shuffle_buffer,
    )
    token_counter = _token_proxy
    if args.tokenizer_path is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
        token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    excluded_ids = _excluded_record_ids(args.exclude_record_id_jsonl)
    rows = collect_rows(
        upstream,
        source,
        token_limit=args.token_limit,
        excluded_record_ids=excluded_ids,
        token_counter=token_counter,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"dataset": args.dataset, "config": args.config, "revision": args.revision, "split": args.split, "records": len(rows), "token_proxy": sum(int(row["token_proxy"]) for row in rows), "excluded_record_ids": len(excluded_ids), "shuffle_seed": args.shuffle_seed, "shuffle_buffer": args.shuffle_buffer, "rights": source.get("rights")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
