from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from benchmark_snapshot_contract import BenchmarkPanel, BenchmarkSnapshotContractError, FrozenBenchmarkRegistry, load_benchmark_snapshot_registry
from development_corpus_admission_contract import (
    AdmissionStatus,
    DevelopmentCorpusAdmissionReport,
    load_admission_registry,
)
from development_corpus_inventory_contract import DevelopmentCorpusInventoryManifest, InventoryStatus, load_inventory_registry
from development_redundancy_gate_contract import (
    DevelopmentRedundancyGateReport,
    RedundancyGateStatus,
    load_redundancy_gate_registry,
)
from development_selection import DevelopmentSelectionStatus, load_development_protocol


class _RedundancyState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    development_ablation_ready: bool
    development_gate_registry: str
    development_gate_registry_sha256: str
    development_gate_registry_file_sha256: str
    development_gate_report: str
    development_gate_report_sha256: str
    development_gate_report_file_sha256: str


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


class _CorpusInventoryState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    registry: str
    manifest: str
    registry_sha256: str
    manifest_sha256: str
    manifest_file_sha256: str
    admission_registry: str
    admission_registry_sha256: str
    admission_registry_file_sha256: str
    admission_report: str
    admission_report_sha256: str
    admission_report_file_sha256: str


class _BlockEightState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    development_corpus_inventory: _CorpusInventoryState


class _ProtocolState(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    benchmark_snapshot_state: _SnapshotPanel
    block_8_implementation: _BlockEightState


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


def _redundancy_blocker(root: Path, state: _RedundancyState) -> tuple[str | None, tuple[Path, ...]]:
    registry_path = root / state.development_gate_registry
    report_path = root / state.development_gate_report
    evidence_paths = tuple(path for path in (registry_path, report_path) if path.is_file())
    if not state.development_ablation_ready:
        return "redundancy_gate_not_ready", evidence_paths
    if not registry_path.is_file() or not report_path.is_file():
        return "redundancy_gate_evidence_invalid", evidence_paths
    try:
        registry = load_redundancy_gate_registry(registry_path)
        report = DevelopmentRedundancyGateReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "redundancy_gate_evidence_invalid", evidence_paths
    identity_valid = (
        registry.identity_sha256() == state.development_gate_registry_sha256
        and _sha256(registry_path) == state.development_gate_registry_file_sha256
        and report.registry_sha256 == state.development_gate_registry_sha256
        and report.report_sha256 == state.development_gate_report_sha256
        and _sha256(report_path) == state.development_gate_report_file_sha256
    )
    evidence_passed = (
        report.status is RedundancyGateStatus.PASSED
        and not report.blocker_codes
        and report.matrix_complete
        and report.inventory_manifest_sha256 == registry.inventory_manifest_sha256
        and report.inventory_manifest_file_sha256 == registry.inventory_manifest_file_sha256
    )
    if not identity_valid or not evidence_passed:
        return "redundancy_gate_evidence_invalid", evidence_paths
    return None, evidence_paths


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
    inventory_state = protocol.block_8_implementation.development_corpus_inventory
    corpus_registry_path = root / inventory_state.registry
    corpus_manifest_path = root / inventory_state.manifest
    admission_registry_path = root / inventory_state.admission_registry
    admission_report_path = root / inventory_state.admission_report
    inventory_paths = tuple(
        path
        for path in (corpus_registry_path, corpus_manifest_path, admission_registry_path, admission_report_path)
        if path.is_file()
    )
    redundancy_blocker, redundancy_paths = _redundancy_blocker(root, redundancy)
    paths = (*base_paths, *benchmark_paths, *inventory_paths, *redundancy_paths)
    blockers: list[str] = []
    if redundancy_blocker is not None:
        blockers.append(redundancy_blocker)
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
    if not all(
        path.is_file()
        for path in (corpus_registry_path, corpus_manifest_path, admission_registry_path, admission_report_path)
    ):
        blockers.append("development_corpus_manifest_missing")
    else:
        try:
            corpus_registry = load_inventory_registry(corpus_registry_path)
            corpus_manifest = DevelopmentCorpusInventoryManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
            admission_registry = load_admission_registry(admission_registry_path)
            admission_report = DevelopmentCorpusAdmissionReport.model_validate_json(admission_report_path.read_text(encoding="utf-8"))
        except ValueError:
            blockers.append("development_corpus_manifest_invalid")
        else:
            identity_valid = (
                corpus_registry.identity_sha256() == inventory_state.registry_sha256
                and corpus_manifest.registry_sha256 == inventory_state.registry_sha256
                and corpus_manifest.manifest_sha256 == inventory_state.manifest_sha256
                and _sha256(corpus_manifest_path) == inventory_state.manifest_file_sha256
                and admission_registry.identity_sha256() == inventory_state.admission_registry_sha256
                and _sha256(admission_registry_path) == inventory_state.admission_registry_file_sha256
                and admission_report.registry_sha256 == inventory_state.admission_registry_sha256
                and admission_report.report_sha256 == inventory_state.admission_report_sha256
                and _sha256(admission_report_path) == inventory_state.admission_report_file_sha256
                and corpus_manifest.admission_report_sha256 == admission_report.report_sha256
            )
            if not identity_valid:
                blockers.append("development_corpus_manifest_invalid")
            elif (
                corpus_manifest.status is not InventoryStatus.ADMITTED
                or admission_report.status is not AdmissionStatus.ADMITTED
                or not admission_report.benchmark_exclusion_complete
                or admission_report.blocker_codes
            ):
                blockers.append("development_corpus_manifest_not_admitted")
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
