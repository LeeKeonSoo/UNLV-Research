from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderContractError(RuntimeError):
    """Raised when a provider transition would reuse invalid evidence."""


class ProviderRole(str, Enum):
    QUALITY = "quality"
    SEMANTIC = "semantic"
    DIAGNOSTIC_VALIDITY = "diagnostic_validity"
    CONTENT_ROUTER = "content_router"


class ProviderLifecycle(str, Enum):
    AUDIT_ONLY = "audit_only"
    CALIBRATED = "calibrated"
    DEVELOPMENT_VALIDATED = "development_validated"
    CONFIRMATORY_VALIDATED = "confirmatory_validated"
    ACTIVE = "active"
    RETIRED = "retired"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class CalibrationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    scope_id: str = Field(min_length=1)


class ValidationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_path: str = Field(min_length=1)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    scope_id: str = Field(min_length=1)
    three_seed_natural_budget_complete: bool


class ProviderSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ProviderRole
    allowed_effect: str = Field(min_length=1)
    provider_output_alone_may_delete: Literal[False]


class ProviderManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    role: ProviderRole
    provider_type: Literal["deterministic", "model", "composite_model"]
    lifecycle: ProviderLifecycle
    artifacts: tuple[FrozenModel, ...]
    tokenizer_id: str | None
    tokenizer_revision: str | None
    normalization: str = Field(min_length=1)
    output_semantics: str = Field(min_length=1)
    implementation_contract_path: str | None = None
    implementation_contract_identity_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    supported_routes: tuple[str, ...]
    supported_languages: tuple[str, ...]
    policy_contribution_authority: bool
    direct_deletion_authority: Literal[False]
    calibration: CalibrationEvidence | None
    validation: ValidationEvidence | None

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "ProviderManifest":
        if self.provider_type != "deterministic" and not self.artifacts:
            raise ProviderContractError("Model providers require at least one frozen artifact")
        if self.provider_type != "deterministic" and (not self.tokenizer_id or not self.tokenizer_revision):
            raise ProviderContractError("Model providers require a frozen tokenizer identity")
        validated = {
            ProviderLifecycle.CALIBRATED,
            ProviderLifecycle.DEVELOPMENT_VALIDATED,
            ProviderLifecycle.CONFIRMATORY_VALIDATED,
            ProviderLifecycle.ACTIVE,
        }
        if self.lifecycle in validated and self.calibration is None:
            raise ProviderContractError("A calibrated or validated provider requires calibration evidence")
        if self.lifecycle in validated and any(artifact.artifact_sha256 is None for artifact in self.artifacts):
            raise ProviderContractError("A calibrated or validated model provider requires artifact hashes")
        if self.lifecycle in {ProviderLifecycle.CONFIRMATORY_VALIDATED, ProviderLifecycle.ACTIVE}:
            if self.validation is None or not self.validation.three_seed_natural_budget_complete:
                raise ProviderContractError("Confirmatory validation requires completed three-seed evidence")
        if self.policy_contribution_authority and self.lifecycle is not ProviderLifecycle.ACTIVE:
            raise ProviderContractError("Only an active provider may contribute to a promoted policy")
        contract_fields = (self.implementation_contract_path, self.implementation_contract_identity_sha256)
        if any(value is not None for value in contract_fields) and not all(
            value is not None for value in contract_fields
        ):
            raise ProviderContractError("Provider implementation contract path and identity must be declared together")
        return self

    def identity_sha256(self) -> str:
        identity = {
            "role": self.role.value,
            "provider_type": self.provider_type,
            "artifacts": [artifact.model_dump(mode="json") for artifact in self.artifacts],
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_revision": self.tokenizer_revision,
            "normalization": self.normalization,
            "output_semantics": self.output_semantics,
            "implementation_contract_path": self.implementation_contract_path,
            "implementation_contract_identity_sha256": self.implementation_contract_identity_sha256,
        }
        encoded = json.dumps(identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ProviderRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["model-provider-registry-v1"]
    status: str = Field(min_length=1)
    framework_model_agnostic: Literal[True]
    replacement_rule: str = Field(min_length=1)
    slots: dict[ProviderRole, ProviderSlot]
    providers: tuple[ProviderManifest, ...]

    @model_validator(mode="after")
    def validate_registry(self) -> "ProviderRegistry":
        if set(self.slots) != set(ProviderRole):
            raise ProviderContractError("Registry must declare every provider role")
        if any(key is not slot.role for key, slot in self.slots.items()):
            raise ProviderContractError("Provider slot keys and roles must match")
        provider_ids = [provider.provider_id for provider in self.providers]
        if len(provider_ids) != len(set(provider_ids)):
            raise ProviderContractError("Provider IDs must be unique")
        return self


def load_provider_registry(path: Path) -> ProviderRegistry:
    return ProviderRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def validate_provider_replacement(current: ProviderManifest, replacement: ProviderManifest) -> None:
    if current.role is not replacement.role:
        raise ProviderContractError("A provider replacement cannot change its role")
    if current.identity_sha256() == replacement.identity_sha256():
        return
    if replacement.lifecycle is not ProviderLifecycle.AUDIT_ONLY or replacement.calibration is not None:
        raise ProviderContractError("A changed provider must restart as audit_only without inherited calibration")
