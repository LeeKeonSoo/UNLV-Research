from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class CommonBaselineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_sha256: Sha256
    record_ids_sha256: Sha256
    source_group_ids: tuple[str, ...] = Field(min_length=1)


class SensitivityArmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1)
    profile_id: Literal["normal", "hard"]
    artifact_sha256: Sha256
    eligible_record_ids_sha256: Sha256
    source_group_ids: tuple[str, ...] = Field(min_length=1)
    common_baseline_sha256: Sha256
    baseline_record_overlap_count: int = Field(ge=0)
    baseline_source_overlap_count: int = Field(ge=0)


class EffectBinSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    route: str = Field(min_length=1)
    bin_id: str = Field(min_length=1)
    rank: int = Field(ge=1)


class EffectBinManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contrastive-effect-bin-manifest-v1"]
    common_baseline_sha256: Sha256
    eligible_record_ids_sha256: Sha256
    bins: tuple[EffectBinSpec, ...] = Field(min_length=1)


class AcceptanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    require_qualified_three_role_provider: Literal[True]
    require_validated_execution_precision: Literal[True]
    require_one_shared_stage_a_baseline: Literal[True]
    require_baseline_disjoint_from_every_arm: Literal[True]
    require_shared_eligible_arm_pool: Literal[True]
    require_ordered_route_effect_bins: Literal[True]
    require_external_natural_budget_evidence: Literal[True]
    require_hard_subset_of_normal: Literal[True]


class SelectorBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_outcomes_available_at_runtime: Literal[False]
    utility_available_at_runtime: Literal[False]
    source_reputation_available_at_runtime: Literal[False]
    runtime_activation_mutation_allowed: Literal[False]


class GateProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contrastive-operating-point-gate-protocol-v1"]
    status: Literal["block_10b_frozen_preflight"]
    contrastive_protocol: EvidenceRef
    contrastive_audit: EvidenceRef
    required_routes: tuple[str, ...] = Field(min_length=1)
    minimum_source_groups_per_route: int = Field(ge=3)
    minimum_ordered_effect_bins_per_route: int = Field(ge=3)
    profile_ids: tuple[Literal["normal", "hard"], ...]
    common_baseline: CommonBaselineSpec | None
    sensitivity_arms: tuple[SensitivityArmSpec, ...]
    effect_bin_manifest: EvidenceRef | None
    external_natural_budget_evidence_sha256: Sha256 | None
    operating_point_artifact_sha256_by_profile: dict[Literal["normal", "hard"], Sha256 | None]
    profile_monotonicity_artifact_sha256: Sha256 | None
    acceptance: AcceptanceSpec
    selector_boundary: SelectorBoundary
    claim_boundary: str = Field(min_length=1)


class RouteAudit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    route: str
    source_group_count: int = Field(ge=0)


class FrozenAudit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["blocked", "ready_for_effect_bin_experiment"]
    route_reports: tuple[RouteAudit, ...]
    blocker_codes: tuple[str, ...]
    benchmark_outcomes_read: Literal[False]
    utility_read: Literal[False]
    runtime_activation: Literal[False]
