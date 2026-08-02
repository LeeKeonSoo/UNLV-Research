from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from framework_objects import CoreId, Lifecycle, ObjectRegistry, StageId, load_framework_objects
from framework_profiles import ProfileRegistry, load_profile_registry
from stage_permissions import StageAuthorityRegistry, StageInputRequest, authorize_stage_input, load_stage_authority

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class RuntimeBridgeError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class LegacyPolicyMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    legacy_policy_id: str = Field(min_length=1)
    v1_policy_id: str = Field(min_length=1)
    disposition: Literal["legacy_compatibility_only"]


class RuntimeBridgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-runtime-bridge-v1"]
    status: Literal["runtime_integrated_block_7"]
    framework_manifest_path: str = Field(min_length=1)
    framework_manifest_sha256: Sha256
    object_registry_path: str = Field(min_length=1)
    object_registry_sha256: Sha256
    profile_registry_path: str = Field(min_length=1)
    profile_registry_sha256: Sha256
    legacy_kernel_path: str = Field(min_length=1)
    legacy_kernel_sha256: Sha256
    legacy_policy_mappings: tuple[LegacyPolicyMapping, ...]
    blocked_v1_policy_ids: tuple[str, ...] = Field(min_length=1)
    new_v1_policy_activation: Literal[False]
    curated_output_equivalence_required: Literal[True]

    @model_validator(mode="after")
    def validate_inventory(self) -> "RuntimeBridgeConfig":
        legacy_ids = {item.legacy_policy_id for item in self.legacy_policy_mappings}
        v1_ids = {item.v1_policy_id for item in self.legacy_policy_mappings}
        if len(legacy_ids) != len(self.legacy_policy_mappings):
            raise RuntimeBridgeError("runtime_bridge_legacy_policy_duplicate")
        if len(v1_ids) != len(self.legacy_policy_mappings):
            raise RuntimeBridgeError("runtime_bridge_v1_policy_duplicate")
        if len(set(self.blocked_v1_policy_ids)) != len(self.blocked_v1_policy_ids):
            raise RuntimeBridgeError("runtime_bridge_blocked_policy_duplicate")
        if not v1_ids <= set(self.blocked_v1_policy_ids):
            raise RuntimeBridgeError("runtime_bridge_compatibility_policy_not_blocked")
        return self


@dataclass(frozen=True, slots=True)
class RuntimeFoundation:
    schema_version: str
    root: Path
    bridge: RuntimeBridgeConfig
    objects: ObjectRegistry
    profiles: ProfileRegistry
    stages: StageAuthorityRegistry


class RuntimeStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: StageId
    core_id: CoreId
    supplied_categories: tuple[str, ...] = Field(min_length=1)


class RuntimeStageTicket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: StageId
    core_id: CoreId
    supplied_categories: tuple[str, ...]
    authorization: Literal["central_stage_permission_granted"]
    selector_decision: None = None


class RuntimeFoundationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-runtime-foundation-report-v1"]
    bridge_status: str
    framework_manifest_sha256: Sha256
    object_registry_sha256: Sha256
    profile_registry_sha256: Sha256
    legacy_kernel_sha256: Sha256
    new_v1_policy_activation: Literal[False]
    blocked_v1_policy_ids: tuple[str, ...]
    stage_tickets: tuple[RuntimeStageTicket, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runtime_foundation(root: Path) -> RuntimeFoundation:
    bridge_path = root / "configs" / "framework_runtime_bridge_v1.json"
    bridge = RuntimeBridgeConfig.model_validate_json(bridge_path.read_text(encoding="utf-8"))
    manifest_path = root / bridge.framework_manifest_path
    object_path = root / bridge.object_registry_path
    profile_path = root / bridge.profile_registry_path
    kernel_path = root / bridge.legacy_kernel_path
    identities = (
        (manifest_path, bridge.framework_manifest_sha256, "runtime_bridge_manifest_identity_mismatch"),
        (object_path, bridge.object_registry_sha256, "runtime_bridge_object_identity_mismatch"),
        (profile_path, bridge.profile_registry_sha256, "runtime_bridge_profile_identity_mismatch"),
        (kernel_path, bridge.legacy_kernel_sha256, "runtime_bridge_kernel_identity_mismatch"),
    )
    for path, expected, reason_code in identities:
        if _sha256(path) != expected:
            raise RuntimeBridgeError(reason_code)
    objects = load_framework_objects(manifest_path, object_path)
    profiles = load_profile_registry(manifest_path, object_path, profile_path)
    lifecycle_by_id = {policy.id: policy.lifecycle for policy in objects.policies}
    for policy_id in bridge.blocked_v1_policy_ids:
        if lifecycle_by_id.get(policy_id) is not Lifecycle.BLOCKED:
            raise RuntimeBridgeError("runtime_bridge_blocked_policy_lifecycle_mismatch")
    return RuntimeFoundation(
        schema_version="framework-runtime-foundation-v1",
        root=root,
        bridge=bridge,
        objects=objects,
        profiles=profiles,
        stages=load_stage_authority(manifest_path),
    )


def authorize_runtime_stage(
    foundation: RuntimeFoundation,
    request: RuntimeStageRequest,
) -> RuntimeStageTicket:
    authorized = authorize_stage_input(
        foundation.stages,
        StageInputRequest(
            stage_id=request.stage_id,
            core_id=request.core_id,
            supplied_categories=request.supplied_categories,
        ),
    )
    return RuntimeStageTicket(
        stage_id=authorized.stage_id,
        core_id=authorized.core_id,
        supplied_categories=authorized.supplied_categories,
        authorization="central_stage_permission_granted",
    )


def build_foundation_report(
    foundation: RuntimeFoundation,
    tickets: tuple[RuntimeStageTicket, ...],
) -> RuntimeFoundationReport:
    bridge = foundation.bridge
    return RuntimeFoundationReport(
        schema_version="framework-runtime-foundation-report-v1",
        bridge_status=bridge.status,
        framework_manifest_sha256=bridge.framework_manifest_sha256,
        object_registry_sha256=bridge.object_registry_sha256,
        profile_registry_sha256=bridge.profile_registry_sha256,
        legacy_kernel_sha256=bridge.legacy_kernel_sha256,
        new_v1_policy_activation=False,
        blocked_v1_policy_ids=bridge.blocked_v1_policy_ids,
        stage_tickets=tickets,
    )


__all__ = [
    "RuntimeStageRequest",
    "RuntimeStageTicket",
    "authorize_runtime_stage",
    "build_foundation_report",
    "load_runtime_foundation",
]
