from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

from contrastive_source_pool_contract import (
    ContrastiveSourcePoolProtocol,
    ContrastiveSourcePoolError,
    PoolRole,
    SourceSpec,
)
from ingestion.candidate_processing import process_candidate

JsonMap: TypeAlias = dict[str, Any]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
TokenCounter: TypeAlias = Callable[[str], int]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: JsonValue) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(encoded.encode())


def _record_id(row: JsonMap, source: SourceSpec, index: int) -> str:
    candidates = (row.get("record_id"), row.get("source_record_id"), row.get("id"))
    value = next((str(item).strip() for item in candidates if item is not None and str(item).strip()), "")
    if not value:
        text = str(row.get(source.text_field) or row.get("text") or "")
        value = f"{index:08d}:{_sha256_bytes(text.encode())}"
    return f"{source.source_group_id}::{value}"


def _stage_a_rows(
    rows: Iterable[JsonMap],
    source: SourceSpec,
    protocol: ContrastiveSourcePoolProtocol,
) -> list[JsonMap]:
    released: list[JsonMap] = []
    for index, row in enumerate(rows):
        candidate = dict(row)
        candidate["record_id"] = _record_id(row, source, index)
        candidate["text"] = str(row.get(source.text_field) or row.get("text") or "")
        processed = process_candidate(
            candidate,
            index=index,
            stage_a_policy=protocol.sampling.stage_a_policy,
        )
        if processed["release_eligibility"]["eligible"]:
            released.append(processed)
    return released


def _input_rows_sha256(rows: list[JsonMap], source: SourceSpec) -> str:
    identities = [
        {
            "record_id": _record_id(row, source, index),
            "text_sha256": _sha256_bytes(
                str(row.get(source.text_field) or row.get("text") or "").encode()
            ),
        }
        for index, row in enumerate(rows)
    ]
    return _sha256_json(identities)


def _stable_sample(
    rows: list[JsonMap],
    source: SourceSpec,
    protocol: ContrastiveSourcePoolProtocol,
) -> list[JsonMap]:
    count = protocol.sampling.records_per_source_after_stage_a
    if len(rows) < count:
        raise ContrastiveSourcePoolError(
            f"stage_a_survivors_below_required_sample:{source.source_id}"
        )
    seed = protocol.sampling.stable_hash_seed
    ranked = sorted(
        rows,
        key=lambda row: _sha256_bytes(
            f"{seed}:{source.source_id}:{row['record_id']}".encode()
        ),
    )
    return ranked[:count]


def _output_row(row: JsonMap, source: SourceSpec, token_counter: TokenCounter) -> JsonMap:
    text = str(row["text"])
    normalized_sha256 = str(row["provenance"]["normalized_sha256"])
    record_uid = _sha256_json(
        {
            "source_group_id": source.source_group_id,
            "stage_a_record_id": row["record_id"],
            "normalized_text_sha256": normalized_sha256,
        }
    )
    return {
        "record_uid": record_uid,
        "text": text,
        "exact_token_count": token_counter(text),
        "contrastive_route": source.route.value,
        "contrastive_source_id": source.source_group_id,
        "pool_role": source.pool_role.value,
        "stage_a_record_id": row["record_id"],
        "normalized_text_sha256": normalized_sha256,
        "stage_a_reason_codes": row["stage_a_decision"]["reason_codes"],
        "source_metadata_selector_visible": False,
    }


def build_source_pools(
    protocol: ContrastiveSourcePoolProtocol,
    rows_by_source_id: Mapping[str, Iterable[JsonMap]],
    token_counter: TokenCounter,
) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    by_role: dict[PoolRole, list[JsonMap]] = {
        PoolRole.COMMON_BASELINE: [],
        PoolRole.ELIGIBLE_ARM: [],
    }
    source_reports: list[JsonMap] = []
    for source in protocol.sources:
        if source.source_id not in rows_by_source_id:
            raise ContrastiveSourcePoolError(f"source_rows_missing:{source.source_id}")
        input_rows = list(rows_by_source_id[source.source_id])
        released = _stage_a_rows(input_rows, source, protocol)
        selected = _stable_sample(released, source, protocol)
        output_rows = [_output_row(row, source, token_counter) for row in selected]
        by_role[source.pool_role].extend(output_rows)
        source_reports.append(
            {
                "source_id": source.source_id,
                "source_group_id": source.source_group_id,
                "route": source.route.value,
                "pool_role": source.pool_role.value,
                "input_record_count": len(input_rows),
                "input_rows_sha256": _input_rows_sha256(input_rows, source),
                "stage_a_survivor_count": len(released),
                "selected_record_count": len(output_rows),
                "selected_exact_tokens": sum(row["exact_token_count"] for row in output_rows),
                "selected_record_ids_sha256": _sha256_json(
                    sorted(row["record_uid"] for row in output_rows)
                ),
            }
        )
    baseline = sorted(by_role[PoolRole.COMMON_BASELINE], key=lambda row: row["record_uid"])
    eligible = sorted(by_role[PoolRole.ELIGIBLE_ARM], key=lambda row: row["record_uid"])
    baseline_ids = {row["record_uid"] for row in baseline}
    eligible_ids = {row["record_uid"] for row in eligible}
    baseline_text = {row["normalized_text_sha256"] for row in baseline}
    eligible_text = {row["normalized_text_sha256"] for row in eligible}
    if baseline_ids & eligible_ids or baseline_text & eligible_text:
        raise ContrastiveSourcePoolError("common_baseline_eligible_pool_overlap")
    manifest: JsonMap = {
        "schema_version": "contrastive-source-pool-materialization-v1",
        "status": "materialized",
        "source_reports": source_reports,
        "common_baseline_record_count": len(baseline),
        "common_baseline_exact_tokens": sum(row["exact_token_count"] for row in baseline),
        "common_baseline_record_ids_sha256": _sha256_json(sorted(baseline_ids)),
        "eligible_pool_record_count": len(eligible),
        "eligible_pool_exact_tokens": sum(row["exact_token_count"] for row in eligible),
        "eligible_pool_record_ids_sha256": _sha256_json(sorted(eligible_ids)),
        "baseline_record_overlap_count": len(baseline_ids & eligible_ids),
        "baseline_normalized_text_overlap_count": len(baseline_text & eligible_text),
        "normal_and_hard_share_eligible_pool": True,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "source_metadata_selector_visible": False,
    }
    manifest["manifest_sha256"] = _sha256_json(manifest)
    return baseline, eligible, manifest


def read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


__all__ = ["build_source_pools", "read_jsonl"]
