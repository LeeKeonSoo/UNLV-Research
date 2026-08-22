from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CoreId(str, Enum):
    VALIDITY = "validity"
    REDUNDANCY = "redundancy"
    QUALITY = "quality"
    COVERAGE = "coverage"


class StageId(str, Enum):
    STAGE_A = "stage_a"
    STAGE_B = "stage_b"
    STAGE_C = "stage_c"


@dataclass(frozen=True, slots=True)
class StagePermissionError(ValueError):
    reason_code: str
    stage_id: StageId
    category: str | None = None

    def __str__(self) -> str:
        suffix = "" if self.category is None else f":{self.category}"
        return f"{self.reason_code}:{self.stage_id.value}{suffix}"


class StageDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: StageId
    position: int = Field(ge=1, le=3)
    scope: str = Field(min_length=1)
    core_ids: tuple[CoreId, ...] = Field(min_length=1)
    allowed_input_categories: tuple[str, ...] = Field(min_length=1)
    forbidden_input_categories: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_categories(self) -> "StageDeclaration":
        if len(set(self.core_ids)) != len(self.core_ids):
            raise StagePermissionError("stage_core_duplicate", self.id)
        if len(set(self.allowed_input_categories)) != len(self.allowed_input_categories):
            raise StagePermissionError("stage_allowed_input_duplicate", self.id)
        if len(set(self.forbidden_input_categories)) != len(self.forbidden_input_categories):
            raise StagePermissionError("stage_forbidden_input_duplicate", self.id)
        if set(self.allowed_input_categories) & set(self.forbidden_input_categories):
            raise StagePermissionError("stage_input_permission_overlap", self.id)
        return self


class StageManifestProjection(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: Literal["curation-framework-v1"]
    stages: tuple[StageDeclaration, ...]
    runtime_forbidden_inputs: tuple[str, ...] = Field(min_length=1)


class StageAuthorityRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: tuple[StageDeclaration, ...]
    runtime_forbidden_inputs: frozenset[str]

    @model_validator(mode="after")
    def validate_topology(self) -> "StageAuthorityRegistry":
        ids = tuple(stage.id for stage in self.stages)
        positions = tuple(stage.position for stage in self.stages)
        if ids != (StageId.STAGE_A, StageId.STAGE_B, StageId.STAGE_C):
            raise StagePermissionError("stage_order_invalid", StageId.STAGE_A)
        if positions != (1, 2, 3):
            raise StagePermissionError("stage_position_invalid", StageId.STAGE_A)
        expected = {
            StageId.STAGE_A: {CoreId.VALIDITY},
            StageId.STAGE_B: {CoreId.REDUNDANCY, CoreId.QUALITY},
            StageId.STAGE_C: {CoreId.COVERAGE},
        }
        for stage in self.stages:
            if set(stage.core_ids) != expected[stage.id]:
                raise StagePermissionError("stage_core_topology_invalid", stage.id)
        return self

    def declaration(self, stage_id: StageId) -> StageDeclaration:
        return next(stage for stage in self.stages if stage.id is stage_id)


class StageInputRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: StageId
    core_id: CoreId
    supplied_categories: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_categories(self) -> "StageInputRequest":
        if len(set(self.supplied_categories)) != len(self.supplied_categories):
            raise StagePermissionError("stage_supplied_input_duplicate", self.stage_id)
        return self


class AuthorizedStageInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: StageId
    core_id: CoreId
    supplied_categories: tuple[str, ...]


def load_stage_authority(path: Path) -> StageAuthorityRegistry:
    projection = StageManifestProjection.model_validate_json(path.read_text(encoding="utf-8"))
    return StageAuthorityRegistry(
        stages=projection.stages,
        runtime_forbidden_inputs=frozenset(projection.runtime_forbidden_inputs),
    )


def authorize_stage_input(
    registry: StageAuthorityRegistry,
    request: StageInputRequest,
) -> AuthorizedStageInput:
    stage = registry.declaration(request.stage_id)
    if request.core_id not in stage.core_ids:
        raise StagePermissionError("stage_core_authority_mismatch", request.stage_id)
    globally_forbidden = registry.runtime_forbidden_inputs
    locally_forbidden = set(stage.forbidden_input_categories)
    allowed = set(stage.allowed_input_categories)
    for category in request.supplied_categories:
        if category in globally_forbidden:
            raise StagePermissionError("stage_runtime_forbidden_input", request.stage_id, category)
        if category in locally_forbidden:
            raise StagePermissionError("stage_local_forbidden_input", request.stage_id, category)
        if category not in allowed:
            raise StagePermissionError("stage_undeclared_input", request.stage_id, category)
    return AuthorizedStageInput(
        stage_id=request.stage_id,
        core_id=request.core_id,
        supplied_categories=request.supplied_categories,
    )


__all__ = [
    "AuthorizedStageInput",
    "CoreId",
    "StageId",
    "StageInputRequest",
    "StagePermissionError",
    "authorize_stage_input",
    "load_stage_authority",
]
