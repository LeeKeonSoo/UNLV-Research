from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from framework_objects import Lifecycle

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


@dataclass(frozen=True, slots=True)
class ContrastiveProtocolError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class ModelRole(str, Enum):
    TARGET = "target"
    QUALITY_REFERENCE = "quality_reference"
    BACKGROUND = "background"


class RoleQualification(str, Enum):
    TARGET_SLM = "target_slm"
    VALIDATED_REFERENCE_POOL = "validated_reference_pool"
    BROAD_BACKGROUND = "broad_background"
    UNQUALIFIED_GENERIC_BASE = "unqualified_generic_base"
    UNASSIGNED = "unassigned"


class Precision(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    INT8 = "int8"
    INT4 = "int4"


class CalibrationStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"


class EvidenceDirection(str, Enum):
    KEEP = "keep_evidence"
    REMOVAL_CANDIDATE = "removal_candidate_only"


class ContrastiveModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ModelRole
    provider_id: str = Field(min_length=1)
    model_id: str | None
    revision: str | None
    artifact_sha256: Sha256 | None
    precision: Precision | None
    quantization_validation_artifact_sha256: Sha256 | None
    role_qualification: RoleQualification
    training_distribution_artifact_sha256: Sha256 | None


class SharedTokenizerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    artifact_sha256: Sha256
    compatibility_artifact_sha256: Sha256
    add_special_tokens: Literal[False]
    append_eos_per_record: Literal[True]


class GapMetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: Literal["learnability_gap", "alignment_gap"]
    formula: str = Field(min_length=1)
    unit: Literal["nats_per_nonpadding_target_token"]
    high_positive_direction: EvidenceDirection


class ContrastiveCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CalibrationStatus
    blocker_codes: tuple[str, ...]
    development_corpus_sha256: Sha256 | None
    sample_count: int | None = Field(default=None, ge=1)
    supported_routes: tuple[str, ...]
    common_baseline_sha256: Sha256 | None
    sensitivity_arm_sha256s: tuple[Sha256, ...]
    disjointness_artifact_sha256: Sha256 | None
    calibration_artifact_sha256: Sha256 | None
    effect_bin_artifact_sha256: Sha256 | None
    external_evidence_sha256: Sha256 | None

    @model_validator(mode="after")
    def validate_ready_evidence(self) -> "ContrastiveCalibration":
        if self.status is CalibrationStatus.BLOCKED:
            if not self.blocker_codes:
                raise ContrastiveProtocolError("contrastive_blocked_without_reason")
            return self
        required = (
            self.development_corpus_sha256,
            self.sample_count,
            self.common_baseline_sha256,
            self.disjointness_artifact_sha256,
            self.calibration_artifact_sha256,
            self.effect_bin_artifact_sha256,
            self.external_evidence_sha256,
        )
        if any(value is None for value in required) or not self.supported_routes:
            raise ContrastiveProtocolError("contrastive_ready_evidence_incomplete")
        if self.blocker_codes:
            raise ContrastiveProtocolError("contrastive_ready_has_blockers")
        if not self.sensitivity_arm_sha256s:
            raise ContrastiveProtocolError("contrastive_sensitivity_arms_missing")
        if len(set(self.sensitivity_arm_sha256s)) != len(self.sensitivity_arm_sha256s):
            raise ContrastiveProtocolError("contrastive_sensitivity_arm_duplicate")
        if self.common_baseline_sha256 in self.sensitivity_arm_sha256s:
            raise ContrastiveProtocolError("contrastive_common_baseline_not_disjoint")
        return self


class ContrastiveQualityProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contrastive-quality-protocol-v2"]
    provider_id: str = Field(min_length=1)
    lifecycle: Lifecycle
    models: tuple[ContrastiveModelSpec, ...]
    tokenizer: SharedTokenizerSpec
    metrics: tuple[GapMetricSpec, ...]
    calibration: ContrastiveCalibration
    unknown_route_action: Literal["abstain_retain"]
    weighted_scalar_emitted: Literal[False]
    runtime_authority: Literal[False]
    direct_deletion_authority: Literal[False]
    replacement_invalidates_calibration: Literal[True]

    @model_validator(mode="after")
    def validate_protocol(self) -> "ContrastiveQualityProtocol":
        by_role = {model.role: model for model in self.models}
        if set(by_role) != set(ModelRole) or len(by_role) != len(self.models):
            raise ContrastiveProtocolError("contrastive_model_roles_incomplete")
        by_metric = {metric.id: metric for metric in self.metrics}
        if set(by_metric) != {"learnability_gap", "alignment_gap"}:
            raise ContrastiveProtocolError("contrastive_metric_inventory_invalid")
        if by_metric["learnability_gap"].high_positive_direction is not EvidenceDirection.KEEP:
            raise ContrastiveProtocolError("contrastive_learnability_direction_invalid")
        if by_metric["alignment_gap"].high_positive_direction is not EvidenceDirection.REMOVAL_CANDIDATE:
            raise ContrastiveProtocolError("contrastive_alignment_direction_invalid")
        if self.calibration.status is CalibrationStatus.READY:
            _validate_ready_models(by_role)
        if self.lifecycle in {Lifecycle.DEVELOPMENT_PASSED, Lifecycle.PROMOTED}:
            if self.calibration.status is not CalibrationStatus.READY:
                raise ContrastiveProtocolError("contrastive_lifecycle_requires_ready_calibration")
        return self

    def identity_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"lifecycle", "calibration"})
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _validate_ready_models(by_role: dict[ModelRole, ContrastiveModelSpec]) -> None:
    target = by_role[ModelRole.TARGET]
    reference = by_role[ModelRole.QUALITY_REFERENCE]
    background = by_role[ModelRole.BACKGROUND]
    if target.role_qualification is not RoleQualification.TARGET_SLM or target.artifact_sha256 is None:
        raise ContrastiveProtocolError("contrastive_target_unqualified")
    if (
        reference.role_qualification is not RoleQualification.VALIDATED_REFERENCE_POOL
        or reference.training_distribution_artifact_sha256 is None
        or reference.artifact_sha256 is None
    ):
        raise ContrastiveProtocolError("contrastive_quality_reference_unqualified")
    if background.role_qualification is not RoleQualification.BROAD_BACKGROUND or background.artifact_sha256 is None:
        raise ContrastiveProtocolError("contrastive_background_unqualified")
    for model in by_role.values():
        if model.precision in {Precision.INT8, Precision.INT4} and model.quantization_validation_artifact_sha256 is None:
            raise ContrastiveProtocolError("contrastive_quantization_unvalidated")


class LossObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_nll: float = Field(ge=0.0)
    quality_reference_nll: float = Field(ge=0.0)
    background_nll: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_finite(self) -> "LossObservation":
        if not all(math.isfinite(value) for value in (self.target_nll, self.quality_reference_nll, self.background_nll)):
            raise ContrastiveProtocolError("contrastive_loss_nonfinite")
        return self


class ContrastiveGaps(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    learnability_gap: float
    alignment_gap: float
    scalar_quality_score: None = None


def compute_contrastive_gaps(observation: LossObservation) -> ContrastiveGaps:
    return ContrastiveGaps(
        learnability_gap=observation.target_nll - observation.quality_reference_nll,
        alignment_gap=observation.quality_reference_nll - observation.background_nll,
    )


def load_contrastive_protocol(path: Path) -> ContrastiveQualityProtocol:
    return ContrastiveQualityProtocol.model_validate_json(path.read_text(encoding="utf-8"))


def validate_protocol_replacement(
    current: ContrastiveQualityProtocol,
    replacement: ContrastiveQualityProtocol,
) -> None:
    if current.identity_sha256() == replacement.identity_sha256():
        return
    if replacement.calibration.status is not CalibrationStatus.BLOCKED:
        raise ContrastiveProtocolError("contrastive_provider_change_requires_recalibration")


__all__ = [
    "ContrastiveProtocolError",
    "ContrastiveQualityProtocol",
    "LossObservation",
    "compute_contrastive_gaps",
    "load_contrastive_protocol",
    "validate_protocol_replacement",
]
