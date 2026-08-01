#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS_ROOT = Path("D:/UNLV-Research/code_5m_corpus_v2")
DEFAULT_CONFIG = ROOT / "configs" / "code_5m_corpus_acquisition_v2.json"
DEFAULT_RAW = DEFAULT_CORPUS_ROOT / "raw_like_candidates.jsonl"
DEFAULT_REFERENCE = (
    DEFAULT_CORPUS_ROOT / "reference_pool_shard_a_v2" / "known_high_quality_raw_records.jsonl",
    DEFAULT_CORPUS_ROOT / "reference_pool_shard_b_v2" / "known_high_quality_raw_records.jsonl",
)
DEFAULT_OUTPUT = DEFAULT_CORPUS_ROOT / "stage0_input"
SELECTOR_VISIBLE_PARTITION_FIELDS = frozenset(
    {"bundle_id", "repository_identity", "path", "change_type", "content_type"}
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_rows(rows: Iterable[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"20260719:{label}:{row['record_id']}".encode("utf-8")).hexdigest(),
    )


def select_reference_rows(
    rows: list[dict[str, Any]], target_tokens: int, token_count: Callable[[dict[str, Any]], int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    collected = 0
    for row in _stable_rows(rows, "reference"):
        selected.append(row)
        collected += token_count(row)
        if collected >= target_tokens:
            break
    if collected < target_tokens:
        raise RuntimeError(f"Reference token target unavailable: {collected} < {target_tokens}")
    return selected


def _license(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else "unknown"
    return str(value or "unknown")


def _source_value(row: dict[str, Any], key: str) -> Any:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    partition = row.get("partition") if isinstance(row.get("partition"), dict) else {}
    return row.get(key) or provenance.get(key) or partition.get(key)


def stage0_candidate(source: dict[str, Any], source_tier: str) -> dict[str, Any]:
    repository = str(
        _source_value(source, "repository_or_origin")
        or _source_value(source, "repository_identity")
        or _source_value(source, "source_name")
        or "unknown"
    )
    source_dataset = str(_source_value(source, "source_dataset") or "github_reference_pool")
    source_uri = str(
        _source_value(source, "source_uri")
        or "https://huggingface.co/datasets/bigcode/the-stack-dedup"
    )
    source_license = _license(_source_value(source, "license"))
    path = str(_source_value(source, "path") or "unknown.py")
    return {
        "record_id": str(source["record_id"]),
        "text": str(source["text"]),
        "provenance": {
            "source_name": repository,
            "source_uri": source_uri,
            "collected_at": str(_source_value(source, "collected_at") or "unknown"),
        },
        "language": {"code": "python", "confidence": 1.0},
        "rights": {"status": "allowed", "license": source_license},
        "pii_context": "repository_code",
        "partition": {
            "split": "train",
            "bundle_id": repository,
            "repository_identity": repository,
            "path": path,
            "change_type": "snapshot",
            "content_type": "code",
            "source_tier": source_tier,
            "source_dataset": source_dataset,
            "source_content_sha256": str(_source_value(source, "content_sha256") or "unknown"),
        },
    }


def _token_counter(tokenizer: Any) -> Callable[[dict[str, Any]], int]:
    def count(row: dict[str, Any]) -> int:
        stored = row.get("token_count_with_eos")
        if isinstance(stored, int) and stored > 0:
            return stored
        return len(tokenizer(str(row["text"]), add_special_tokens=False).input_ids) + 1

    return count


def _write_input(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"records": rows}, ensure_ascii=False), encoding="utf-8")


def build(
    config_path: Path,
    raw_path: Path,
    reference_paths: Iterable[Path],
    output_dir: Path,
    *,
    allow_download: bool,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = load_json(config_path)
    tokenizer_id = str(config["target_model"]["tokenizer_id"])
    target_tokens = int(config["target"]["reference_source_target_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=not allow_download, use_fast=True)
    token_count = _token_counter(tokenizer)
    raw = _read_jsonl(raw_path)
    references = [row for path in reference_paths for row in _read_jsonl(path)]
    selected_reference = select_reference_rows(references, target_tokens, token_count)
    candidates = [
        *[stage0_candidate(row, "raw_like") for row in raw],
        *[stage0_candidate(row, "known_high_quality_reference") for row in selected_reference],
    ]
    record_ids = [str(row["record_id"]) for row in candidates]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate record IDs after raw/reference materialization")
    input_path = output_dir / "stage0_raw_candidates.json"
    _write_input(input_path, candidates)
    report = {
        "schema_version": "code-5m-stage0-input-report-v1",
        "status": "code_5m_stage0_input_frozen",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "summary": {
            "raw_like_record_count": len(raw),
            "reference_available_record_count": len(references),
            "reference_selected_record_count": len(selected_reference),
            "reference_selected_tokens_with_eos": sum(token_count(row) for row in selected_reference),
            "candidate_record_count": len(candidates),
        },
        "source_paths": {"raw_like": str(raw_path), "reference": [str(path) for path in reference_paths]},
        "stage_b_isolation": {
            "source_tier_available_to_stage_b": False,
            "source_dataset_available_to_stage_b": False,
            "utility_available_to_stage_b": False,
            "benchmark_outcomes_available_to_stage_b": False,
        },
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_dir / "stage0_input_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze 5M raw/reference Stage-0 input.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--reference", type=Path, action="append", dest="references")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = build(args.config, args.raw, args.references or DEFAULT_REFERENCE, args.output_dir, allow_download=args.allow_download)
    print(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
