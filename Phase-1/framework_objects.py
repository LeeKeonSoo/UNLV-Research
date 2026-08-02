from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class FrameworkObjectError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class CoreId(str, Enum):
    VALIDITY = "validity"
    REDUNDANCY = "redundancy"
    QUALITY = "quality"
    COVERAGE = "coverage"


class StageId(str, Enum):
    STAGE_A = "stage_a"
    STAGE_B = "stage_b"
    STAGE_C = "stage_c"


class Lifecycle(str, Enum):
    CANDIDATE = "candidate"
    DEVELOPMENT_PASSED = "development_passed"
    PROMOTED = "promoted"
    BLOCKED = "blocked"
    RETIRED = "retired"


class DecisionAuthority(str, Enum):
    QUARANTINE_OR_REMOVE = "quarantine_or_remove"
    REMOVE_NONREPRESENTATIVE = "remove_nonrepresentative"
    QUALITY_DECISION = "quality_decision"
    MATERIALIZATION_VETO = "materialization_veto"


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class ThresholdProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str = Field(min_length=1)
    comparison_direction: str = Field(min_length=1)
    derivation_procedure: str = Field(min_length=1)
    development_corpus_sha256: Sha256
    sample_count: int = Field(ge=1)
    supported_routes: tuple[str, ...] = Field(min_length=1)
    provider_identity_sha256: Sha256
    tokenizer_identity_sha256: Sha256
    uncertainty_procedure: str = Field(min_length=1)
    fixture_artifact_sha256: Sha256
    ablation_artifact_sha256: Sha256
    external_evidence_sha256: Sha256
    lifecycle: Lifecycle
    invalidation_conditions: tuple[str, ...] = Field(min_length=1)


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    implementation_path: str = Field(min_length=1)
    implementation_sha256: Sha256
    deterministic: bool


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    core_id: CoreId
    method_id: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    semantics: str = Field(min_length=1)
    provider_id: str | None
    threshold: ThresholdProvenance | None


class PolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    core_id: CoreId
    stage_id: StageId
    metric_ids: tuple[str, ...] = Field(min_length=1)
    lifecycle: Lifecycle
    decision_authority: DecisionAuthority
    reason_codes: tuple[str, ...] = Field(min_length=1)
    forbidden_inputs: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[EvidenceReference, ...]


class ProviderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    lifecycle: Lifecycle
    output_metric_ids: tuple[str, ...] = Field(min_length=1)
    identity_sha256: Sha256
    direct_deletion_authority: Literal[False]
    replacement_invalidates_calibration: Literal[True]


class ObjectRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-objects-v1"]
    status: Literal["design_only_until_block_7"]
    framework_manifest_path: str = Field(min_length=1)
    framework_manifest_sha256: Sha256
    methods: tuple[MethodSpec, ...] = Field(min_length=1)
    metrics: tuple[MetricSpec, ...] = Field(min_length=1)
    policies: tuple[PolicySpec, ...] = Field(min_length=1)
    providers: tuple[ProviderSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lineage(self) -> "ObjectRegistry":
        methods = _unique(self.methods, "framework_method_id_duplicate")
        metrics = _unique(self.metrics, "framework_metric_id_duplicate")
        providers = _unique(self.providers, "framework_provider_id_duplicate")
        _unique(self.policies, "framework_policy_id_duplicate")
        for metric in self.metrics:
            if metric.method_id not in methods:
                raise FrameworkObjectError("framework_metric_method_missing")
            if metric.provider_id is not None and metric.provider_id not in providers:
                raise FrameworkObjectError("framework_metric_provider_missing")
        expected_stage = {
            CoreId.VALIDITY: StageId.STAGE_A,
            CoreId.REDUNDANCY: StageId.STAGE_B,
            CoreId.QUALITY: StageId.STAGE_B,
            CoreId.COVERAGE: StageId.STAGE_C,
        }
        expected_authority = {
            CoreId.VALIDITY: DecisionAuthority.QUARANTINE_OR_REMOVE,
            CoreId.REDUNDANCY: DecisionAuthority.REMOVE_NONREPRESENTATIVE,
            CoreId.QUALITY: DecisionAuthority.QUALITY_DECISION,
            CoreId.COVERAGE: DecisionAuthority.MATERIALIZATION_VETO,
        }
        for policy in self.policies:
            if policy.stage_id is not expected_stage[policy.core_id]:
                raise FrameworkObjectError("framework_policy_stage_core_mismatch")
            if policy.decision_authority is not expected_authority[policy.core_id]:
                raise FrameworkObjectError("framework_policy_authority_core_mismatch")
            for metric_id in policy.metric_ids:
                if metric_id not in metrics:
                    raise FrameworkObjectError("framework_policy_metric_missing")
                if metrics[metric_id].core_id is not policy.core_id:
                    raise FrameworkObjectError("framework_policy_metric_core_mismatch")
        for provider in self.providers:
            if any(metric_id not in metrics for metric_id in provider.output_metric_ids):
                raise FrameworkObjectError("framework_provider_metric_missing")
        return self


def _unique(items: tuple[MethodSpec, ...] | tuple[MetricSpec, ...] | tuple[PolicySpec, ...] | tuple[ProviderSpec, ...], reason_code: str):
    by_id = {item.id: item for item in items}
    if len(by_id) != len(items):
        raise FrameworkObjectError(reason_code)
    return by_id


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_framework_objects(manifest_path: Path, registry_source: Path | ObjectRegistry) -> ObjectRegistry:
    registry = (
        registry_source
        if isinstance(registry_source, ObjectRegistry)
        else ObjectRegistry.model_validate_json(registry_source.read_text(encoding="utf-8"))
    )
    if registry.framework_manifest_sha256 != _sha256(manifest_path):
        raise FrameworkObjectError("framework_manifest_identity_mismatch")
    root = manifest_path.parent.parent
    for method in registry.methods:
        if _sha256(root / method.implementation_path) != method.implementation_sha256:
            raise FrameworkObjectError(f"framework_method_identity_mismatch:{method.id}")
    return registry


__all__ = ["FrameworkObjectError", "ObjectRegistry", "load_framework_objects"]
