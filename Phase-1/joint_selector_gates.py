from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from joint_selector_contract import JointGateBundle, JointGateOrigin, JointProfileName, load_joint_profiles


class RedundancyGateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    runtime_activation: bool


class QualityGateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    runtime_activation: bool
    all_registered_routes_empirically_ready: bool


class CoverageGateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    runtime_activation: bool | Literal["development_and_confirmatory_only"]
    all_required_views_empirically_ready: bool


class SplitGateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    benchmark_feedback_to_runtime_allowed: bool


class ProtocolGateConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    split_and_leakage_contract: SplitGateConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_current_joint_gates(root: Path) -> JointGateBundle:
    redundancy_path = root / "configs" / "redundancy_v2.json"
    quality_path = root / "configs" / "quality_effect_engine_v2.json"
    coverage_path = root / "configs" / "coverage_engine_v2.json"
    profile_path = root / "configs" / "joint_selector_profiles_v1.json"
    protocol_path = root / "protocols" / "target_aware_core_completion_v1.json"
    redundancy = RedundancyGateConfig.model_validate_json(redundancy_path.read_text(encoding="utf-8"))
    quality = QualityGateConfig.model_validate_json(quality_path.read_text(encoding="utf-8"))
    coverage = CoverageGateConfig.model_validate_json(coverage_path.read_text(encoding="utf-8"))
    protocol = ProtocolGateConfig.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    hard = load_joint_profiles(profile_path).by_name(JointProfileName.HARD)
    return JointGateBundle(
        origin=JointGateOrigin.FROZEN_REGISTRY,
        redundancy_ready=redundancy.runtime_activation,
        quality_ready=quality.runtime_activation and quality.all_registered_routes_empirically_ready,
        coverage_ready=(
            coverage.runtime_activation in (True, "development_and_confirmatory_only")
            and coverage.all_required_views_empirically_ready
        ),
        hard_extension_ready=hard.hard_extension_frozen and bool(hard.hard_extension_policy_ids),
        external_results_hidden=not protocol.split_and_leakage_contract.benchmark_feedback_to_runtime_allowed,
        evidence_artifact_hashes=tuple(_sha256(path) for path in (redundancy_path, quality_path, coverage_path)),
    )
