from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from framework_objects import Lifecycle, load_framework_objects

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class ProfileContractError(ValueError):
    reason_code: str
    offending_ids: tuple[str, ...] = ()

    def __str__(self) -> str:
        suffix = "" if not self.offending_ids else f":{','.join(self.offending_ids)}"
        return f"{self.reason_code}{suffix}"


class ProfileId(str, Enum):
    NORMAL = "normal"
    HARD = "hard"


class PolicyLifecycleSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    lifecycle: Lifecycle


class ProfileSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ProfileId
    inherits_profile: Literal["normal"] | None
    policy_ids: tuple[str, ...] = Field(min_length=1)
    additional_policy_ids: tuple[str, ...]
    threshold_overrides: tuple[str, ...]
    release_enabled: bool
    fixed_retention_fraction_allowed: Literal[False]
    maximum_token_budget_allowed: Literal[False]
    output: Literal["full_reason_coded_curated_pool"]

    @model_validator(mode="after")
    def validate_unique_policies(self) -> "ProfileSpec":
        if len(set(self.policy_ids)) != len(self.policy_ids):
            raise ProfileContractError("profile_policy_duplicate")
        if len(set(self.additional_policy_ids)) != len(self.additional_policy_ids):
            raise ProfileContractError("profile_additional_policy_duplicate")
        if self.threshold_overrides:
            raise ProfileContractError("profile_threshold_override_forbidden")
        return self


class ProfileRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-profiles-v1"]
    status: Literal["runtime_integrated_block_7"]
    framework_manifest_sha256: Sha256
    object_registry_sha256: Sha256
    policy_lifecycles: tuple[PolicyLifecycleSnapshot, ...]
    profiles: tuple[ProfileSpec, ...]
    blocker_codes: tuple[str, ...]
    retained_set_invariant: Literal["hard_subset_or_equal_normal"]

    @model_validator(mode="after")
    def validate_composition(self) -> "ProfileRegistry":
        lifecycle_by_id = {item.policy_id: item.lifecycle for item in self.policy_lifecycles}
        if len(lifecycle_by_id) != len(self.policy_lifecycles):
            raise ProfileContractError("profile_policy_lifecycle_duplicate")
        by_id = {profile.id: profile for profile in self.profiles}
        if set(by_id) != set(ProfileId) or len(by_id) != len(self.profiles):
            raise ProfileContractError("profile_public_inventory_invalid")
        normal = by_id[ProfileId.NORMAL]
        hard = by_id[ProfileId.HARD]
        if normal.inherits_profile is not None or normal.additional_policy_ids:
            raise ProfileContractError("profile_normal_inheritance_invalid")
        if hard.inherits_profile != ProfileId.NORMAL.value:
            raise ProfileContractError("profile_hard_inheritance_invalid")
        normal_ids = set(normal.policy_ids)
        hard_ids = set(hard.policy_ids)
        if not normal_ids < hard_ids:
            raise ProfileContractError("profile_hard_policy_set_not_strict_superset")
        if set(hard.additional_policy_ids) != hard_ids - normal_ids:
            raise ProfileContractError("profile_hard_additional_policy_mismatch")
        referenced = normal_ids | hard_ids
        missing = tuple(sorted(referenced - set(lifecycle_by_id)))
        if missing:
            raise ProfileContractError("profile_policy_lifecycle_missing", missing)
        for profile in self.profiles:
            unpromoted = tuple(
                policy_id
                for policy_id in profile.policy_ids
                if lifecycle_by_id[policy_id] is not Lifecycle.PROMOTED
            )
            if profile.release_enabled and unpromoted:
                raise ProfileContractError("profile_release_contains_unpromoted_policy", unpromoted)
        has_unpromoted = any(lifecycle is not Lifecycle.PROMOTED for lifecycle in lifecycle_by_id.values())
        expected_blockers = ("profile_contains_unpromoted_policy",) if has_unpromoted else ()
        if self.blocker_codes != expected_blockers:
            raise ProfileContractError("profile_blocker_inventory_mismatch")
        return self


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_profile_registry(
    manifest_path: Path,
    object_registry_path: Path,
    profile_registry_path: Path,
) -> ProfileRegistry:
    registry = ProfileRegistry.model_validate_json(profile_registry_path.read_text(encoding="utf-8"))
    if registry.framework_manifest_sha256 != _sha256(manifest_path):
        raise ProfileContractError("profile_framework_manifest_identity_mismatch")
    if registry.object_registry_sha256 != _sha256(object_registry_path):
        raise ProfileContractError("profile_object_registry_identity_mismatch")
    objects = load_framework_objects(manifest_path, object_registry_path)
    observed = {policy.id: policy.lifecycle for policy in objects.policies}
    declared = {item.policy_id: item.lifecycle for item in registry.policy_lifecycles}
    if declared != observed:
        raise ProfileContractError("profile_policy_lifecycle_snapshot_mismatch")
    return registry


def validate_retained_set_monotonicity(
    normal_retained: tuple[str, ...],
    hard_retained: tuple[str, ...],
) -> None:
    hard_only = tuple(sorted(set(hard_retained) - set(normal_retained)))
    if hard_only:
        raise ProfileContractError("profile_hard_retained_set_not_subset", hard_only)


__all__ = [
    "ProfileContractError",
    "ProfileRegistry",
    "load_profile_registry",
    "validate_retained_set_monotonicity",
]
