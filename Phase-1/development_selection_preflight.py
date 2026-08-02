from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from benchmark_snapshot_contract import BenchmarkPanel, BenchmarkSnapshotContractError, FrozenBenchmarkRegistry, load_benchmark_snapshot_registry
from development_selection import DevelopmentSelectionStatus, load_development_protocol


class _RedundancyState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    development_ablation_ready: bool


class _QualityState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    all_registered_routes_empirically_ready: bool


class _CoverageState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    all_required_views_empirically_ready: bool


class _SnapshotState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    status: str
    source_registry: str | None = None
    frozen_manifest: str | None = None
    source_registry_sha256: str | None = None
    frozen_manifest_sha256: str | None = None
    frozen_manifest_file_sha256: str | None = None
    task_count: int | None = None


class _SnapshotPanel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    code: _SnapshotState
    math: _SnapshotState
    general: _SnapshotState


class _ProtocolState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    benchmark_snapshot_state: _SnapshotPanel


class CurrentDevelopmentPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-selection-v1-current-preflight"]
    status: DevelopmentSelectionStatus
    profiles_frozen: Literal[False]
    blocker_codes: tuple[str, ...]
    evidence_artifact_hashes: tuple[str, ...]
    manifest_sha256: str
    benchmark_outcomes_read: Literal[False] = False
    confirmatory_outcomes_read: Literal[False] = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_blocker(root: Path, panel_name: str, panel: _SnapshotState) -> str | None:
    if panel.status != "frozen":
        return f"{panel_name}_benchmark_snapshot_not_frozen"
    if panel_name == "code":
        return None
    if None in (
        panel.source_registry,
        panel.frozen_manifest,
        panel.source_registry_sha256,
        panel.frozen_manifest_sha256,
        panel.frozen_manifest_file_sha256,
        panel.task_count,
    ):
        return f"{panel_name}_benchmark_snapshot_contract_invalid"
    registry_path = root / str(panel.source_registry)
    manifest_path = root / str(panel.frozen_manifest)
    if not registry_path.is_file() or not manifest_path.is_file():
        return f"{panel_name}_benchmark_snapshot_artifact_missing"
    try:
        registry = load_benchmark_snapshot_registry(registry_path)
        frozen = FrozenBenchmarkRegistry.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (BenchmarkSnapshotContractError, OSError, ValueError):
        return f"{panel_name}_benchmark_snapshot_contract_invalid"
    expected_panel = BenchmarkPanel(panel_name)
    task_count = sum(item.task_count for item in frozen.snapshots if item.panel is expected_panel)
    if (
        registry.identity_sha256() != panel.source_registry_sha256
        or frozen.source_registry_sha256 != panel.source_registry_sha256
        or frozen.manifest_sha256 != panel.frozen_manifest_sha256
        or _sha256(manifest_path) != panel.frozen_manifest_file_sha256
        or task_count != panel.task_count
    ):
        return f"{panel_name}_benchmark_snapshot_contract_invalid"
    return None


def evaluate_current_development_preflight(root: Path) -> CurrentDevelopmentPreflight:
    development_path = root / "configs" / "development_selection_v1.json"
    development = load_development_protocol(development_path)
    hard_inventory_path = root / development.hard_candidate_inventory
    base_paths = (
        development_path,
        root / "configs" / "redundancy_v2.json",
        root / "configs" / "quality_effect_engine_v2.json",
        root / "configs" / "coverage_engine_v2.json",
        root / "protocols" / "target_aware_core_completion_v1.json",
        hard_inventory_path,
    )
    redundancy = _RedundancyState.model_validate_json(base_paths[1].read_text(encoding="utf-8"))
    quality = _QualityState.model_validate_json(base_paths[2].read_text(encoding="utf-8"))
    coverage = _CoverageState.model_validate_json(base_paths[3].read_text(encoding="utf-8"))
    protocol = _ProtocolState.model_validate_json(base_paths[4].read_text(encoding="utf-8"))
    benchmark_paths = tuple(
        root / value
        for value in (
            protocol.benchmark_snapshot_state.math.source_registry,
            protocol.benchmark_snapshot_state.math.frozen_manifest,
        )
        if value is not None
    )
    paths = (*base_paths, *benchmark_paths)
    blockers: list[str] = []
    if not redundancy.development_ablation_ready:
        blockers.append("redundancy_gate_not_ready")
    if not quality.all_registered_routes_empirically_ready:
        blockers.append("quality_gate_not_ready")
    if not coverage.all_required_views_empirically_ready:
        blockers.append("coverage_gate_not_ready")
    for panel_name, panel in (("code", protocol.benchmark_snapshot_state.code), ("math", protocol.benchmark_snapshot_state.math), ("general", protocol.benchmark_snapshot_state.general)):
        blocker = _snapshot_blocker(root, panel_name, panel)
        if blocker is not None:
            blockers.append(blocker)
    if _sha256(hard_inventory_path) != development.hard_candidate_inventory_sha256:
        blockers.append("hard_candidate_inventory_hash_mismatch")
    if not (root / "configs" / "development_corpus_manifest_v1.json").is_file():
        blockers.append("development_corpus_manifest_missing")
    evidence = tuple(_sha256(path) for path in paths)
    payload = {"blockers": sorted(blockers), "evidence": evidence}
    manifest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return CurrentDevelopmentPreflight(
        schema_version="development-selection-v1-current-preflight",
        status=DevelopmentSelectionStatus.BLOCKED,
        profiles_frozen=False,
        blocker_codes=tuple(sorted(blockers)),
        evidence_artifact_hashes=evidence,
        manifest_sha256=manifest,
    )


__all__ = ["CurrentDevelopmentPreflight", "evaluate_current_development_preflight"]
