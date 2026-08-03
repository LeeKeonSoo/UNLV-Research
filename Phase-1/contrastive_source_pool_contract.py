from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
UtcTimestamp = Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]


@dataclass(frozen=True, slots=True)
class ContrastiveSourcePoolError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class Route(str, Enum):
    CODE = "code_artifact"
    MATH = "mathematical_content"
    GENERAL = "general_prose"


class PoolRole(str, Enum):
    COMMON_BASELINE = "common_baseline"
    ELIGIBLE_ARM = "eligible_arm"


class LocationKind(str, Enum):
    LOCAL_JSONL = "local_jsonl"
    HUGGINGFACE_FILE = "huggingface_file"


class SourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    source_group_id: str = Field(min_length=1)
    route: Route
    pool_role: PoolRole
    location_kind: LocationKind
    local_path: str | None = None
    expected_file_sha256: Sha256 | None = None
    dataset_id: str | None = None
    revision: GitRevision | None = None
    loader: Literal["json", "parquet"] | None = None
    data_file: str | None = None
    text_field: str = Field(min_length=1)
    stable_record_id_field: str | None = None
    source_license_field: str | None = None
    allowed_source_licenses: tuple[str, ...] = ()
    declared_license: str | None = None
    exact_token_collection_target: int | None = Field(default=None, gt=0)
    required_text_route: Route | None = None
    collection_output: str | None = None

    @model_validator(mode="after")
    def validate_location(self) -> "SourceSpec":
        local = (self.local_path, self.expected_file_sha256)
        remote = (self.dataset_id, self.revision, self.loader, self.data_file)
        if self.location_kind is LocationKind.LOCAL_JSONL:
            if any(value is None for value in local) or any(value is not None for value in remote):
                raise ContrastiveSourcePoolError("local_source_location_incomplete")
        elif any(value is None for value in remote) or any(value is not None for value in local):
            raise ContrastiveSourcePoolError("huggingface_source_location_incomplete")
        if self.location_kind is LocationKind.HUGGINGFACE_FILE:
            if self.exact_token_collection_target is None:
                raise ContrastiveSourcePoolError("remote_collection_target_missing")
            if self.source_license_field is None and self.declared_license is None:
                raise ContrastiveSourcePoolError("remote_source_license_contract_missing")
            if self.collection_output is None:
                raise ContrastiveSourcePoolError("remote_collection_output_missing")
        elif self.collection_output is not None:
            raise ContrastiveSourcePoolError("local_source_must_not_declare_collection_output")
        if self.required_text_route is not None and self.required_text_route is not self.route:
            raise ContrastiveSourcePoolError("text_route_filter_must_match_source_route")
        return self


class SamplingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_a_policy: Literal["text_only_v2"]
    exact_tokenizer_id: Literal["Qwen/Qwen3-4B-Base"]
    exact_tokenizer_revision: GitRevision
    records_per_source_after_stage_a: int = Field(ge=100)
    stable_hash_seed: str = Field(min_length=1)
    common_baseline_output: str = Field(min_length=1)
    eligible_pool_output: str = Field(min_length=1)


class BoundarySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_metadata_selector_visible: Literal[False]
    benchmark_outcomes_read: Literal[False]
    utility_read: Literal[False]
    normal_and_hard_share_eligible_record_ids: Literal[True]
    baseline_disjoint_from_eligible_records_and_sources: Literal[True]
    effect_bins_are_separate_arms: Literal[False]


class ContrastiveSourcePoolProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contrastive-operating-point-source-pool-v1"]
    status: Literal["preregistered_before_collection_and_scoring"]
    collection_timestamp_utc: UtcTimestamp
    sources: tuple[SourceSpec, ...]
    sampling: SamplingSpec
    boundary: BoundarySpec
    confirmatory_source_group_ids: tuple[str, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_closed_source_matrix(self) -> "ContrastiveSourcePoolProtocol":
        if len({item.source_id for item in self.sources}) != len(self.sources):
            raise ContrastiveSourcePoolError("source_ids_not_unique")
        if len({item.source_group_id for item in self.sources}) != len(self.sources):
            raise ContrastiveSourcePoolError("source_group_ids_not_unique")
        expected_routes = set(Route)
        if {item.route for item in self.sources} != expected_routes:
            raise ContrastiveSourcePoolError("route_matrix_incomplete")
        for route in Route:
            route_sources = [item for item in self.sources if item.route is route]
            roles = [item.pool_role for item in route_sources]
            if roles.count(PoolRole.COMMON_BASELINE) != 1 or roles.count(PoolRole.ELIGIBLE_ARM) != 2:
                raise ContrastiveSourcePoolError(f"source_role_matrix_incomplete:{route.value}")
        baseline_groups = {
            item.source_group_id for item in self.sources if item.pool_role is PoolRole.COMMON_BASELINE
        }
        eligible_groups = {
            item.source_group_id for item in self.sources if item.pool_role is PoolRole.ELIGIBLE_ARM
        }
        if baseline_groups & eligible_groups:
            raise ContrastiveSourcePoolError("baseline_eligible_source_overlap")
        if (baseline_groups | eligible_groups) & set(self.confirmatory_source_group_ids):
            raise ContrastiveSourcePoolError("development_confirmatory_source_overlap")
        return self


def load_source_pool_protocol(path: Path) -> ContrastiveSourcePoolProtocol:
    return ContrastiveSourcePoolProtocol.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["ContrastiveSourcePoolError", "ContrastiveSourcePoolProtocol", "LocationKind", "PoolRole", "Route", "SourceSpec", "load_source_pool_protocol"]
