from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

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


def evaluate_current_development_preflight(root: Path) -> CurrentDevelopmentPreflight:
    development_path = root / "configs" / "development_selection_v1.json"
    development = load_development_protocol(development_path)
    hard_inventory_path = root / development.hard_candidate_inventory
    paths = (
        development_path,
        root / "configs" / "redundancy_v2.json",
        root / "configs" / "quality_effect_engine_v2.json",
        root / "configs" / "coverage_engine_v2.json",
        root / "protocols" / "target_aware_core_completion_v1.json",
        hard_inventory_path,
    )
    redundancy = _RedundancyState.model_validate_json(paths[1].read_text(encoding="utf-8"))
    quality = _QualityState.model_validate_json(paths[2].read_text(encoding="utf-8"))
    coverage = _CoverageState.model_validate_json(paths[3].read_text(encoding="utf-8"))
    protocol = _ProtocolState.model_validate_json(paths[4].read_text(encoding="utf-8"))
    blockers: list[str] = []
    if not redundancy.development_ablation_ready:
        blockers.append("redundancy_gate_not_ready")
    if not quality.all_registered_routes_empirically_ready:
        blockers.append("quality_gate_not_ready")
    if not coverage.all_required_views_empirically_ready:
        blockers.append("coverage_gate_not_ready")
    for panel_name, panel in (("code", protocol.benchmark_snapshot_state.code), ("math", protocol.benchmark_snapshot_state.math), ("general", protocol.benchmark_snapshot_state.general)):
        if panel.status != "frozen":
            blockers.append(f"{panel_name}_benchmark_snapshot_not_frozen")
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
