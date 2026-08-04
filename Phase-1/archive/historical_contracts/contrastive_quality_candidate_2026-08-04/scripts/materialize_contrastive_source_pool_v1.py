#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_router import route_content
from contrastive_source_pool_contract import (
    ContrastiveSourcePoolError,
    LocationKind,
    SourceSpec,
    load_source_pool_protocol,
)
from contrastive_source_pool_materialization import build_source_pools, read_jsonl
from scripts.collect_huggingface_text_candidate_pool import collect_rows

JsonMap = dict[str, Any]


def _load_runtime_dependencies(importer: Any = importlib.import_module) -> tuple[Any, Any]:
    datasets_module = importer("datasets")
    transformers_module = importer("transformers")
    return datasets_module.load_dataset, transformers_module.AutoTokenizer


def _remote_url(source: SourceSpec) -> str:
    return (
        f"https://huggingface.co/datasets/{source.dataset_id}/resolve/"
        f"{source.revision}/{source.data_file}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_rows(source: SourceSpec) -> list[JsonMap]:
    path = Path(str(source.local_path))
    if _sha256_file(path) != source.expected_file_sha256:
        raise ContrastiveSourcePoolError(f"local_source_hash_mismatch:{source.source_id}")
    return read_jsonl(path)


def _route_filter(rows: Iterable[JsonMap], source: SourceSpec) -> Iterable[JsonMap]:
    if source.required_text_route is None:
        return rows
    return (
        row
        for row in rows
        if source.required_text_route.value
        in route_content(str(row.get(source.text_field) or ""))["route_labels"]
    )


def _remote_rows(
    source: SourceSpec,
    timestamp: str,
    token_counter: Any,
    load_dataset: Any,
) -> list[JsonMap]:
    upstream = load_dataset(
        source.loader,
        data_files=_remote_url(source),
        split="train",
        streaming=True,
    ).shuffle(seed=31082026, buffer_size=10_000)
    manifest: JsonMap = {
        "source_name": source.source_group_id,
        "source_uri": _remote_url(source),
        "collected_at": timestamp,
        "text_field": source.text_field,
        "stable_record_id_field": source.stable_record_id_field,
        "source_license_field": source.source_license_field,
        "allowed_source_licenses": list(source.allowed_source_licenses)
        if source.allowed_source_licenses
        else None,
        "rights": {
            "status": "allowed",
            "license": source.declared_license or "per-record declared license",
        },
        "pii_context": "general",
        "language": {"code": "und", "confidence": None},
        "partition": {"source_pool_role": source.pool_role.value, "content_type": source.route.value},
    }
    if manifest["allowed_source_licenses"] is None:
        manifest.pop("allowed_source_licenses")
    rows = collect_rows(
        _route_filter(upstream, source),
        manifest,
        token_limit=int(source.exact_token_collection_target or 0),
        token_counter=token_counter,
    )
    output = Path(str(source.collection_output))
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, rows)
    return rows


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the frozen Block 10C source pools.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "protocols" / "contrastive_operating_point_source_pool_v2.json",
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("D:/UNLV-Research/contrastive_quality_v2/development/source_pool_manifest_v1.json"),
    )
    args = parser.parse_args()
    protocol = load_source_pool_protocol(args.protocol)
    load_dataset, auto_tokenizer = _load_runtime_dependencies()
    tokenizer = auto_tokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    token_counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    rows_by_source: dict[str, Iterable[JsonMap]] = {}
    for source in protocol.sources:
        rows_by_source[source.source_id] = (
            _local_rows(source)
            if source.location_kind is LocationKind.LOCAL_JSONL
            else _remote_rows(
                source,
                protocol.collection_timestamp_utc,
                token_counter,
                load_dataset,
            )
        )
    baseline, eligible, manifest = build_source_pools(protocol, rows_by_source, token_counter)
    baseline_path = Path(protocol.sampling.common_baseline_output)
    eligible_path = Path(protocol.sampling.eligible_pool_output)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(baseline_path, baseline)
    _write_jsonl(eligible_path, eligible)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline_records": len(baseline), "eligible_records": len(eligible)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
