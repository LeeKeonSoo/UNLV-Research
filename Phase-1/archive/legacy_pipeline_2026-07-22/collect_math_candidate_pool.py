#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from curation_artifacts import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
DEFAULT_CONFIG = Path("configs") / "math_collection_contract.json"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "math_raw_mixed_5m_collection_report.json"
DEFAULT_MARKDOWN = OUTPUT_DIR / "validation" / "math_raw_mixed_5m_collection_report.md"


def load_collection_config(path: Path) -> JsonMap:
    config = load_json(path)
    if config.get("status") != "frozen_before_collection":
        raise RuntimeError(f"Unexpected collection status: {config.get('status')}")
    return config


def token_proxy(text: str) -> int:
    return len(text.split())


def passes_lexical_quarantine(text: str, config: JsonMap) -> bool:
    normalization = config["normalization_contract"]
    minimum = int(normalization["minimum_token_proxy"])
    blocked = tuple(str(value).lower() for value in normalization["blocked_benchmark_terms"])
    lowered = text.lower()
    return token_proxy(text) >= minimum and not any(term in lowered for term in blocked)


def _text_from_row(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    return "\n\n".join(str(row[field]).strip() for field in fields if row.get(field))


def _stream(dataset_id: str, split: str):
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split, streaming=True)


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def collect_pool(config: JsonMap, pool_name: str, target_override: int | None) -> JsonMap:
    spec = config[pool_name]
    target = int(target_override or spec["target_token_proxy"])
    dataset_id = str(spec["dataset_id"])
    split = str(spec["split"])
    text_fields = tuple(str(field) for field in spec["text_fields"])
    max_records = int(spec["max_records"])
    selected: list[JsonMap] = []
    selected_tokens = 0
    skipped = 0
    for row_index, source in enumerate(_stream(dataset_id, split)):
        text = _text_from_row(source, text_fields)
        if not passes_lexical_quarantine(text, config):
            skipped += 1
            continue
        row_tokens = token_proxy(text)
        selected.append(
            {
                "record_uid": f"math-raw-mixed-v1::{pool_name}::{len(selected):08d}",
                "domain": "math",
                "pool_role": str(spec["role"]),
                "source_dataset_id": dataset_id,
                "source_split": split,
                "source_row_index": row_index,
                "text": text,
                "token_proxy": row_tokens,
            }
        )
        selected_tokens += row_tokens
        if selected_tokens >= target or len(selected) >= max_records:
            break
    return {
        "rows": selected,
        "requested_token_proxy": target,
        "collected_token_proxy": selected_tokens,
        "records": len(selected),
        "skipped_lexical_quarantine_or_short": skipped,
        "target_reached": selected_tokens >= target,
        "max_records_reached": len(selected) >= max_records,
        "dataset_id": dataset_id,
        "split": split,
        "pool_role": str(spec["role"]),
    }


def _pool_report(pool: JsonMap, path: Path) -> JsonMap:
    return {
        **{key: value for key, value in pool.items() if key != "rows"},
        "path": str(path),
        "sha256": sha256_file(path),
    }


def build(config_path: Path, raw_target: int | None, reference_target: int | None) -> JsonMap:
    os.environ.setdefault("HF_HOME", "D:\\hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "D:\\hf_cache\\datasets")
    config = load_collection_config(config_path)
    output_dir = Path(str(config["output_dir"]))
    raw = collect_pool(config, "raw_pool", raw_target)
    reference = collect_pool(config, "reference_pool", reference_target)
    raw_path = output_dir / "raw_mixed_math_candidates.jsonl"
    reference_path = output_dir / "math_reference_context.jsonl"
    _write_jsonl(raw_path, raw["rows"])
    _write_jsonl(reference_path, reference["rows"])
    report = {
        "schema_version": "math-raw-mixed-collection-report-v1",
        "status": "math_raw_mixed_collection_complete" if raw["target_reached"] else "math_raw_mixed_collection_incomplete",
        "claim_boundary": config["claim_boundary"],
        "collection_config": str(config_path),
        "raw_pool": _pool_report(raw, raw_path),
        "reference_pool": _pool_report(reference, reference_path),
        "benchmark_quarantine": config["benchmark_quarantine"],
        "forbidden_uses": config["forbidden_uses"],
        "next_action": "Run Stage-0 contamination and risk quarantine before Stage-A chunking or Stage-B selection.",
        "source_sha256": {str(config_path): sha256_file(config_path)},
    }
    save_json(DEFAULT_REPORT, report)
    DEFAULT_MARKDOWN.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_MARKDOWN.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    raw = report["raw_pool"]
    reference = report["reference_pool"]
    return "\n".join(
        [
            "# Math Raw-Mixed Collection",
            "",
            f"Status: `{report['status']}`",
            "",
            "| Pool | Records | Token proxy | Requested | Target reached |",
            "| --- | ---: | ---: | ---: | --- |",
            f"| Raw mixed | {raw['records']} | {raw['collected_token_proxy']} | {raw['requested_token_proxy']} | {raw['target_reached']} |",
            f"| Reference context | {reference['records']} | {reference['collected_token_proxy']} | {reference['requested_token_proxy']} | {reference['target_reached']} |",
            "",
            f"Next: {report['next_action']}",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect frozen raw-mixed Math pools.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-target", type=int)
    parser.add_argument("--reference-target", type=int)
    args = parser.parse_args()
    report = build(args.config, args.raw_target, args.reference_target)
    print(json.dumps({"status": report["status"], "raw_pool": report["raw_pool"]}, ensure_ascii=False))
    return 0 if report["raw_pool"]["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
