from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
Revision = Annotated[str, StringConstraints(pattern=REVISION_RE.pattern)]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
PositiveInt = Annotated[int, Field(gt=0)]
type JsonValue = str | int | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshotContractError(RuntimeError):
    reason_code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.reason_code}: {self.detail}"


class BenchmarkPanel(str, Enum):
    MATH = "math"
    GENERAL = "general"


class BenchmarkAdapter(str, Enum):
    GSM8K = "gsm8k"
    HENDRYCKS_MATH = "hendrycks_math"
    MMLU_PRO = "mmlu_pro"
    BBH = "bbh"
    ARC_CHALLENGE = "arc_challenge"
    HELLASWAG = "hellaswag"


class BenchmarkSnapshotSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str
    panel: BenchmarkPanel
    repository_id: str
    revision: Revision
    adapter: BenchmarkAdapter
    declared_split: str
    source_files: tuple[str, ...]

    def validate_contract(self) -> None:
        if self.benchmark_id != self.adapter.value or not self.repository_id or not self.declared_split:
            raise BenchmarkSnapshotContractError("benchmark_identity_invalid", "Benchmark identity, repository, and split are required")
        if not self.source_files or len(set(self.source_files)) != len(self.source_files):
            raise BenchmarkSnapshotContractError("benchmark_source_files_invalid", "Source files must be nonempty and unique")
        if any(_unsafe_relative_path(value) for value in self.source_files):
            raise BenchmarkSnapshotContractError("benchmark_source_path_unsafe", "Source files must be safe cache-relative paths")


class BenchmarkSnapshotRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["benchmark-snapshot-registry-v1"]
    status: Literal["development_exclusion_snapshot_only"]
    benchmark_outcomes_available: Literal[False]
    selector_membership_mutation_allowed: Literal[False]
    source_reputation_used: Literal[False]
    snapshots: tuple[BenchmarkSnapshotSpec, ...]

    @model_validator(mode="after")
    def validate_complete_registry(self) -> "BenchmarkSnapshotRegistry":
        adapters = tuple(item.adapter for item in self.snapshots)
        if len(adapters) != len(set(adapters)) or set(adapters) != set(BenchmarkAdapter):
            raise BenchmarkSnapshotContractError("benchmark_registry_incomplete", "All six adapters are required exactly once")
        for item in self.snapshots:
            item.validate_contract()
            expected_panel = BenchmarkPanel.MATH if item.adapter in (BenchmarkAdapter.GSM8K, BenchmarkAdapter.HENDRYCKS_MATH) else BenchmarkPanel.GENERAL
            if item.panel is not expected_panel:
                raise BenchmarkSnapshotContractError("benchmark_panel_mismatch", "Benchmark adapter is assigned to the wrong panel")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class SourceFileSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str
    sha256: Sha256
    row_count: PositiveInt


class FrozenBenchmarkSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str
    panel: BenchmarkPanel
    repository_id: str
    revision: Revision
    declared_split: str
    source_files: tuple[SourceFileSnapshot, ...]
    task_count: PositiveInt
    unique_text_hash_count: PositiveInt
    canonical_snapshot_filename: str
    canonical_snapshot_sha256: Sha256


class FrozenBenchmarkRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["frozen-benchmark-snapshot-registry-v1"]
    status: Literal["frozen_for_development_corpus_exclusion"]
    source_registry_sha256: Sha256
    snapshots: tuple[FrozenBenchmarkSnapshot, ...]
    manifest_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    selector_membership_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_frozen_registry(self) -> "FrozenBenchmarkRegistry":
        benchmark_ids = tuple(item.benchmark_id for item in self.snapshots)
        if len(benchmark_ids) != len(set(benchmark_ids)) or set(benchmark_ids) != {item.value for item in BenchmarkAdapter}:
            raise BenchmarkSnapshotContractError("frozen_benchmark_registry_incomplete", "All six frozen benchmark snapshots are required exactly once")
        for item in self.snapshots:
            expected_panel = BenchmarkPanel.MATH if item.benchmark_id in (BenchmarkAdapter.GSM8K.value, BenchmarkAdapter.HENDRYCKS_MATH.value) else BenchmarkPanel.GENERAL
            if item.panel is not expected_panel:
                raise BenchmarkSnapshotContractError("frozen_benchmark_panel_mismatch", "A frozen benchmark is assigned to the wrong panel")
        payload = {
            "source_registry_sha256": self.source_registry_sha256,
            "snapshots": [item.model_dump(mode="json") for item in self.snapshots],
        }
        if self.manifest_sha256 != hash_json(payload):
            raise BenchmarkSnapshotContractError("frozen_benchmark_manifest_hash_mismatch", "Frozen benchmark metadata does not reproduce its manifest hash")
        return self


def _unsafe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return path.is_absolute() or ".." in path.parts or not value.endswith(".parquet")


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_benchmark_snapshot_registry(path: Path) -> BenchmarkSnapshotRegistry:
    return BenchmarkSnapshotRegistry.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "BenchmarkAdapter", "BenchmarkPanel", "BenchmarkSnapshotContractError",
    "BenchmarkSnapshotRegistry", "BenchmarkSnapshotSpec", "FrozenBenchmarkRegistry",
    "FrozenBenchmarkSnapshot", "SourceFileSnapshot", "hash_json",
    "load_benchmark_snapshot_registry",
]
