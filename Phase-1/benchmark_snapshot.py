from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated, assert_never

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from benchmark_snapshot_contract import (
    BenchmarkAdapter,
    BenchmarkSnapshotContractError,
    BenchmarkSnapshotRegistry,
    BenchmarkSnapshotSpec,
    FrozenBenchmarkRegistry,
    FrozenBenchmarkSnapshot,
    SourceFileSnapshot,
    hash_json,
)


class _GsmRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    question: str
    answer: str


class _MathRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    problem: str
    solution: str


class _MmluRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    question_id: int
    question: str
    options: tuple[str, ...]
    answer: str
    cot_content: str


class _BbhRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    input: str
    target: str


class _ArcChoices(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    text: tuple[str, ...]
    label: tuple[str, ...]


class _ArcRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str
    question: str
    choices: _ArcChoices
    answerKey: str


class _HellaRow(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    ind: int
    source_id: str
    ctx: str
    endings: tuple[str, ...]
    label: str


class _SnapshotRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    segments: tuple[str, ...]
    normalized_text_sha256: str


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _record(task_id: str, segments: tuple[str, ...]) -> _SnapshotRecord:
    normalized = tuple(item for item in (_normalize(value) for value in segments) if item)
    if not task_id or not normalized:
        raise BenchmarkSnapshotContractError("benchmark_task_payload_missing", "Every benchmark task needs an ID and textual payload")
    digest = hashlib.sha256("\n".join(normalized).encode()).hexdigest()
    return _SnapshotRecord(task_id=task_id, segments=normalized, normalized_text_sha256=digest)


def _rows(spec: BenchmarkSnapshotSpec, path: Path) -> tuple[_SnapshotRecord, ...]:
    raw_rows = pq.read_table(path).to_pylist()
    records: list[_SnapshotRecord] = []
    for index, raw in enumerate(raw_rows):
        match spec.adapter:
            case BenchmarkAdapter.GSM8K:
                row = _GsmRow.model_validate(raw)
                record = _record(str(index), (row.question, row.answer))
            case BenchmarkAdapter.HENDRYCKS_MATH:
                row = _MathRow.model_validate(raw)
                record = _record(f"{path.parent.name}/{index}", (row.problem, row.solution))
            case BenchmarkAdapter.MMLU_PRO:
                row = _MmluRow.model_validate(raw)
                record = _record(str(row.question_id), (row.question, *row.options, row.cot_content, row.answer))
            case BenchmarkAdapter.BBH:
                row = _BbhRow.model_validate(raw)
                record = _record(f"{path.parent.name}/{index}", (row.input, row.target))
            case BenchmarkAdapter.ARC_CHALLENGE:
                row = _ArcRow.model_validate(raw)
                choices = tuple(f"{label}: {text}" for label, text in zip(row.choices.label, row.choices.text, strict=True))
                record = _record(row.id, (row.question, *choices, row.answerKey))
            case BenchmarkAdapter.HELLASWAG:
                row = _HellaRow.model_validate(raw)
                record = _record(f"{row.source_id}/{row.ind}", (row.ctx, *row.endings, row.label))
            case unreachable:
                assert_never(unreachable)
        records.append(record)
    return tuple(records)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_snapshot(spec: BenchmarkSnapshotSpec, cache_root: Path, output_root: Path) -> FrozenBenchmarkSnapshot:
    source_paths = tuple(cache_root / value for value in spec.source_files)
    records = tuple(record for path in source_paths for record in _rows(spec, path))
    task_ids = tuple(record.task_id for record in records)
    if not records or len(task_ids) != len(set(task_ids)):
        raise BenchmarkSnapshotContractError("benchmark_task_identity_invalid", "Snapshot tasks must be nonempty and uniquely identified")
    ordered = tuple(sorted(records, key=lambda item: item.task_id))
    output = output_root / f"{spec.benchmark_id}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in ordered:
            handle.write(record.model_dump_json() + "\n")
    sources = tuple(
        SourceFileSnapshot(relative_path=relative, sha256=_sha256_file(path), row_count=pq.ParquetFile(path).metadata.num_rows)
        for relative, path in zip(spec.source_files, source_paths, strict=True)
    )
    return FrozenBenchmarkSnapshot(
        benchmark_id=spec.benchmark_id,
        panel=spec.panel,
        repository_id=spec.repository_id,
        revision=spec.revision,
        declared_split=spec.declared_split,
        source_files=sources,
        task_count=len(ordered),
        unique_text_hash_count=len({item.normalized_text_sha256 for item in ordered}),
        canonical_snapshot_filename=output.name,
        canonical_snapshot_sha256=_sha256_file(output),
    )


def build_benchmark_registry(registry: BenchmarkSnapshotRegistry, cache_root: Path, output_root: Path) -> FrozenBenchmarkRegistry:
    missing = tuple(relative for spec in registry.snapshots for relative in spec.source_files if not (cache_root / relative).is_file())
    if missing:
        raise BenchmarkSnapshotContractError("benchmark_source_file_missing", ", ".join(missing))
    snapshots = tuple(_build_snapshot(spec, cache_root, output_root) for spec in registry.snapshots)
    payload = {"source_registry_sha256": registry.identity_sha256(), "snapshots": [item.model_dump(mode="json") for item in snapshots]}
    return FrozenBenchmarkRegistry(
        schema_version="frozen-benchmark-snapshot-registry-v1",
        status="frozen_for_development_corpus_exclusion",
        source_registry_sha256=registry.identity_sha256(),
        snapshots=snapshots,
        manifest_sha256=hash_json(payload),
    )


__all__ = ["build_benchmark_registry"]
