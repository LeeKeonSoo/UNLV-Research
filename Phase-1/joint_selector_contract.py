from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, assert_never

from pydantic import BaseModel, ConfigDict, model_validator

from coverage_contract import (
    CoverageChunk,
    CoverageDecision,
    CoverageRequest,
    CoverageStratum,
    FrozenSimilarity,
    RepresentativeFamily,
)
from quality_effect_calibration import EffectDirection
from quality_effect_engine import QualityEffectDecision, QualityEffectDecisionName


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_AWARE_POLICY_IDS = (
    "candidate_redundancy_v2",
    "stage_c_calibrated_quality_effect_candidate",
    "stage_c_coverage_support_candidate",
)


class JointSelectorContractError(RuntimeError):
    """Raised when joint-selection evidence violates the frozen contract."""


class JointProfileName(str, Enum):
    BASE = "base"
    NORMAL = "normal"
    HARD = "hard"


class JointSelectionStatus(str, Enum):
    BASE_MATERIALIZED = "base_materialized"
    CANDIDATE_MATERIALIZED = "candidate_materialized"
    BLOCKED_RETAIN_BASE = "blocked_retain_base"


class JointGateOrigin(str, Enum):
    CONTRACT_FIXTURE = "contract_fixture"
    FROZEN_REGISTRY = "frozen_registry"


class JointProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: JointProfileName
    profile_id: str
    status: str
    development_only: Literal[True]
    required_policy_ids: tuple[str, ...]
    hard_extension_policy_ids: tuple[str, ...]
    hard_extension_frozen: bool
    requires_hard_extension: bool

    def identity_sha256(self) -> str:
        encoded = json.dumps(self.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class JointProfileRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["joint-selector-profiles-v1"]
    status: str
    runtime_activation: Literal[False]
    post_run_override_allowed: Literal[False]
    benchmark_feedback_allowed: Literal[False]
    source_selection_axis: Literal[False]
    target_retention_fraction_allowed: Literal[False]
    hidden_token_budget_allowed: Literal[False]
    existing_active_profile_preserved: Literal["normal_structural_v1"]
    profiles: tuple[JointProfileSpec, ...]

    @model_validator(mode="after")
    def validate_profile_structure(self) -> "JointProfileRegistry":
        expected_ids = {
            JointProfileName.BASE: "base_target_aware_v1",
            JointProfileName.NORMAL: "normal_target_aware_v1",
            JointProfileName.HARD: "hard_target_aware_v1",
        }
        for profile in self.profiles:
            if profile.profile_id != expected_ids[profile.name]:
                raise JointSelectorContractError("Joint profile IDs cannot be reassigned")
            match profile.name:
                case JointProfileName.BASE:
                    valid = (
                        not profile.required_policy_ids
                        and not profile.hard_extension_policy_ids
                        and not profile.hard_extension_frozen
                        and not profile.requires_hard_extension
                    )
                case JointProfileName.NORMAL:
                    valid = (
                        profile.required_policy_ids == TARGET_AWARE_POLICY_IDS
                        and not profile.hard_extension_policy_ids
                        and not profile.hard_extension_frozen
                        and not profile.requires_hard_extension
                    )
                case JointProfileName.HARD:
                    valid = (
                        profile.required_policy_ids == TARGET_AWARE_POLICY_IDS
                        and profile.requires_hard_extension
                        and profile.hard_extension_frozen == bool(profile.hard_extension_policy_ids)
                    )
                case unreachable:
                    assert_never(unreachable)
            if not valid:
                raise JointSelectorContractError("Base, Normal, and Hard policy structure is immutable")
        return self

    def by_name(self, name: JointProfileName) -> JointProfileSpec:
        profile = next((item for item in self.profiles if item.name is name), None)
        if profile is None:
            raise JointSelectorContractError(f"Missing joint profile: {name.value}")
        return profile


def load_joint_profiles(path: Path) -> JointProfileRegistry:
    registry = JointProfileRegistry.model_validate_json(path.read_text(encoding="utf-8"))
    names = tuple(profile.name for profile in registry.profiles)
    if len(names) != len(set(names)) or set(names) != set(JointProfileName):
        raise JointSelectorContractError("Joint profiles must define Base, Normal, and Hard exactly once")
    return registry


@dataclass(frozen=True, slots=True)
class JointGateBundle:
    origin: JointGateOrigin
    redundancy_ready: bool
    quality_ready: bool
    coverage_ready: bool
    hard_extension_ready: bool
    external_results_hidden: bool
    evidence_artifact_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.evidence_artifact_hashes) < 3:
            raise JointSelectorContractError("Joint gates require separate Redundancy, Quality, and Coverage artifacts")
        if len(self.evidence_artifact_hashes) != len(set(self.evidence_artifact_hashes)):
            raise JointSelectorContractError("Joint gate artifacts must be unique")
        if any(not SHA256_RE.fullmatch(artifact) for artifact in self.evidence_artifact_hashes):
            raise JointSelectorContractError("Joint gates require lowercase SHA-256 artifacts")

    def identity_sha256(self) -> str:
        payload = {
            "origin": self.origin.value,
            "redundancy_ready": self.redundancy_ready,
            "quality_ready": self.quality_ready,
            "coverage_ready": self.coverage_ready,
            "hard_extension_ready": self.hard_extension_ready,
            "external_results_hidden": self.external_results_hidden,
            "evidence_artifact_hashes": self.evidence_artifact_hashes,
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class JointSelectionRequest:
    chunks: tuple[CoverageChunk, ...]
    redundancy_families: tuple[RepresentativeFamily, ...]
    quality_decisions: tuple[QualityEffectDecision, ...]
    coverage_strata: tuple[CoverageStratum, ...]
    similarities: tuple[FrozenSimilarity, ...]
    quality_provider_identity_sha256: str
    semantic_provider_id: str
    semantic_provider_identity_sha256: str

    def __post_init__(self) -> None:
        universe = frozenset(chunk.uid for chunk in self.chunks)
        decision_ids = tuple(decision.chunk_uid for decision in self.quality_decisions)
        if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != universe:
            raise JointSelectorContractError("Every universe chunk requires exactly one Quality decision")
        if not SHA256_RE.fullmatch(self.quality_provider_identity_sha256):
            raise JointSelectorContractError("Joint requests require a frozen Quality-provider identity")
        for decision in self.quality_decisions:
            if decision.may_mutate_curated_membership or decision.benchmark_outcomes_read or decision.utility_read:
                raise JointSelectorContractError("Quality evidence cannot mutate or read external outcomes")
            if decision.evidence_artifact_hashes and self.quality_provider_identity_sha256 not in decision.evidence_artifact_hashes:
                raise JointSelectorContractError("Quality evidence must trace to the declared provider identity")
            match decision.decision:
                case QualityEffectDecisionName.REJECT_CANDIDATE:
                    valid_reject = (
                        decision.reason_code == "quality_nonpositive_effect_supported"
                        and decision.effect_direction is EffectDirection.SUPPORTED_NONPOSITIVE
                        and len(decision.evidence_artifact_hashes) == 5
                    )
                    if not valid_reject:
                        raise JointSelectorContractError("Quality rejection requires the complete calibrated trace")
                case QualityEffectDecisionName.ELIGIBLE_KEEP | QualityEffectDecisionName.ABSTAIN_RETAIN:
                    pass
                case unreachable:
                    assert_never(unreachable)
        CoverageRequest(
            chunks=self.chunks,
            proposed_survivors=universe,
            strata=self.coverage_strata,
            redundancy_families=self.redundancy_families,
            similarities=self.similarities,
            exclusions=(),
            provider_id=self.semantic_provider_id,
            provider_identity_sha256=self.semantic_provider_identity_sha256,
        )


@dataclass(frozen=True, slots=True)
class JointRemovalTrace:
    chunk_uid: str
    authority_core: Literal["redundancy", "quality"]
    policy_id: str
    reason_code: str
    representative_chunk_uid: str | None
    evidence_artifact_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JointSelectionResult:
    profile_name: JointProfileName
    profile_sha256: str
    status: JointSelectionStatus
    reason_code: str
    selected_uids: tuple[str, ...]
    removal_traces: tuple[JointRemovalTrace, ...]
    coverage_decision: CoverageDecision | None
    input_sha256: str
    manifest_sha256: str
    policy_artifact_hashes: tuple[str, ...]
    evidence_artifact_hashes: tuple[str, ...]
    model_provider_identity_hashes: tuple[str, ...]
    coverage_required_retain_uids: tuple[str, ...] = ()
    coverage_rematerialization_applied: bool = False
    may_mutate_active_runtime: bool = False
    benchmark_outcomes_read: bool = False
    utility_read: bool = False
    source_identity_read: bool = False
    target_retention_fraction_read: bool = False
    post_run_override_applied: bool = False
