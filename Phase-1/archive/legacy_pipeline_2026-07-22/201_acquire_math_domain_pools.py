#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

CONFIG_PATH = Path("configs") / "math_domain_block4_acquisition_freeze_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "math_domain_block4"
REPORT_PATH = OUTPUT_DIR / "validation" / "math_domain_block4_acquisition_report.json"
MD_REPORT_PATH = OUTPUT_DIR / "validation" / "math_domain_block4_acquisition_report.md"


def _token_proxy(text: str) -> int:
    return len(text.split())


def _blocked_terms(config: JsonMap) -> tuple[str, ...]:
    terms = config["normalization_contract"]["blocked_benchmark_terms"]
    return tuple(str(term).lower() for term in terms)


def _passes_filter(text: str, config: JsonMap) -> bool:
    lowered = text.lower()
    minimum = int(config["normalization_contract"]["minimum_token_proxy"])
    return _token_proxy(text) >= minimum and not any(term in lowered for term in _blocked_terms(config))


def _row_text(row: Mapping[str, JsonValue], fields: Iterable[str]) -> str:
    parts = [str(row[field]).strip() for field in fields if row.get(field)]
    return "\n\n".join(parts)


def _write_jsonl(path: Path, rows: list[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _stream_rows(dataset_id: str, split: str) -> Iterable[Mapping[str, JsonValue]]:
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split, streaming=True)


def _collect_pool(config: JsonMap, pool_name: str, text_fields: tuple[str, ...]) -> tuple[list[JsonMap], int]:
    pool = config[pool_name]
    dataset_id = str(pool["dataset_id"])
    split = str(pool["split"])
    target = int(config["target_records_per_pool"])
    rows: list[JsonMap] = []
    skipped = 0
    for index, source in enumerate(_stream_rows(dataset_id, split)):
        text = _row_text(source, text_fields)
        if not _passes_filter(text, config):
            skipped += 1
            continue
        rows.append(
            {
                "record_uid": f"math::{pool_name}::{len(rows):06d}",
                "pool": pool_name,
                "domain": "math",
                "source_dataset_id": dataset_id,
                "source_split": split,
                "text": text,
                "token_proxy": _token_proxy(text),
                "metadata": {
                    "source_row_index": index,
                    "source": str(source.get("source", "")),
                    "role": str(pool["role"]),
                },
            }
        )
        if len(rows) >= target:
            break
    return rows, skipped


def build() -> JsonMap:
    os.environ.setdefault("HF_HOME", "D:\\hf_cache")
    os.environ.setdefault("HF_DATASETS_CACHE", "D:\\hf_cache\\datasets")
    config = load_json(CONFIG_PATH)
    output_dir = Path(str(config.get("output_dir", DEFAULT_OUTPUT_DIR)))
    raw_rows, raw_skipped = _collect_pool(config, "raw_mixed_pool", ("text",))
    reference_rows, reference_skipped = _collect_pool(config, "known_high_quality_reference_pool", ("problem", "solution"))

    raw_path = output_dir / "raw_mixed_math_pool.jsonl"
    reference_path = output_dir / "known_high_quality_math_reference_pool.jsonl"
    _write_jsonl(raw_path, raw_rows)
    _write_jsonl(reference_path, reference_rows)

    report = {
        "schema_version": "math-domain-block4-acquisition-report-v1",
        "status": "math_domain_block4_acquisition_pools_ready",
        "freeze_config": str(CONFIG_PATH),
        "utility_scope": config["utility_scope"],
        "pools": {
            "raw_mixed_pool": {
                "path": str(raw_path),
                "records": len(raw_rows),
                "token_proxy": sum(int(row["token_proxy"]) for row in raw_rows),
                "source_dataset_id": config["raw_mixed_pool"]["dataset_id"],
                "sha256": sha256_file(raw_path),
                "blocked_or_short_records_skipped": raw_skipped,
            },
            "known_high_quality_reference_pool": {
                "path": str(reference_path),
                "records": len(reference_rows),
                "token_proxy": sum(int(row["token_proxy"]) for row in reference_rows),
                "source_dataset_id": config["known_high_quality_reference_pool"]["dataset_id"],
                "sha256": sha256_file(reference_path),
                "blocked_or_short_records_skipped": reference_skipped,
            },
        },
        "stage_c_benchmark_quarantine": config["stage_c_benchmarks"],
        "stage_materialization_status": config["stage_materialization_status"],
        "next_actions": config["next_actions"],
        "source_sha256": {str(CONFIG_PATH): sha256_file(CONFIG_PATH)},
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    raw = report["pools"]["raw_mixed_pool"]
    reference = report["pools"]["known_high_quality_reference_pool"]
    return "\n".join(
        [
            "# Math Domain Block 4 Acquisition",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Pools",
            "",
            f"- Raw mixed pool: `{raw['records']}` records, `{raw['token_proxy']}` token proxy",
            f"- Reference pool: `{reference['records']}` records, `{reference['token_proxy']}` token proxy",
            "",
            "## Stage Materialization",
            "",
            f"Status: `{report['stage_materialization_status']}`",
            "",
        ]
    )


def main() -> int:
    report = build()
    print(f"[math-domain-block4-acquisition] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
