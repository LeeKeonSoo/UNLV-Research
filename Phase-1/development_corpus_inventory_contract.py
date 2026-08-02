from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
type JsonValue = str | int | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DevelopmentCorpusInventoryError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class InventoryDomain(str, Enum):
    CODE = "code"
    MATH = "math"
    GENERAL = "general"


class SourceRole(str, Enum):
    CLEAN_CONTROL = "clean_control"
    RAW_LIKE = "raw_like"


class ConfirmatoryReference(str, Enum):
    FROZEN = "frozen"
    PENDING = "pending"


class ScenarioOrigin(str, Enum):
    OBSERVED = "observed"
    METAMORPHIC = "metamorphic"


class SliceStatus(str, Enum):
    INVENTORIED = "inventoried"
    MATERIALIZED = "materialized"
    MATERIALIZATION_PENDING = "materialization_pending"


class InventoryStatus(str, Enum):
    ADMITTED = "admitted"
    BLOCKED = "blocked"


class InventoryAdmissionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    report_sha256: Sha256
    benchmark_exclusion_complete: bool
    frozen_confirmatory_domains: tuple[InventoryDomain, ...]
    blocker_codes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_domain_evidence(self) -> "InventoryAdmissionEvidence":
        if len(self.frozen_confirmatory_domains) != len(set(self.frozen_confirmatory_domains)):
            raise DevelopmentCorpusInventoryError("frozen_confirmatory_domains_not_unique")
        if self.benchmark_exclusion_complete and "benchmark_contamination_detected" in self.blocker_codes:
            raise DevelopmentCorpusInventoryError("benchmark_completion_evidence_mismatch")
        return self


class InventorySourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    domain: InventoryDomain
    role: SourceRole
    path: str
    id_fields: tuple[str, ...]
    text_field: str
    expected_file_sha256: Sha256
    selector_visible_source_metadata: Literal[False]


class DevelopmentCorpusInventoryRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-corpus-inventory-registry-v1"]
    status: Literal["block-8-inventory-only"]
    normalization: Literal["unicode-nfkc-whitespace-collapse-v1"]
    output_root: str
    parent_records_per_slice: PositiveInt
    sources: tuple[InventorySourceSpec, ...]
    confirmatory_references: dict[InventoryDomain, ConfirmatoryReference]
    metamorphic_transformations: dict[str, tuple[str, ...]]
    benchmark_outcomes_available: Literal[False]
    selector_membership_mutation_allowed: Literal[False]

    @model_validator(mode="after")
    def validate_closed_inventory(self) -> "DevelopmentCorpusInventoryRegistry":
        expected = {(domain, role) for domain in InventoryDomain for role in SourceRole}
        observed = {(item.domain, item.role) for item in self.sources}
        if observed != expected or len(self.sources) != len(expected):
            raise DevelopmentCorpusInventoryError("inventory_source_matrix_incomplete")
        if len({item.source_id for item in self.sources}) != len(self.sources):
            raise DevelopmentCorpusInventoryError("inventory_source_ids_not_unique")
        if set(self.confirmatory_references) != set(InventoryDomain):
            raise DevelopmentCorpusInventoryError("confirmatory_reference_state_incomplete")
        expected_transformations = {"duplicate_heavy", "malformed", "boilerplate_heavy"}
        if set(self.metamorphic_transformations) != expected_transformations or any(not value for value in self.metamorphic_transformations.values()):
            raise DevelopmentCorpusInventoryError("metamorphic_transformations_incomplete")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class InventorySourceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str
    domain: InventoryDomain
    role: SourceRole
    file_sha256: Sha256
    record_count: PositiveInt
    unique_record_id_count: PositiveInt
    unique_normalized_text_count: PositiveInt
    stored_normalized_hash_mismatch_count: NonNegativeInt


class DomainPairEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: InventoryDomain
    clean_raw_record_id_overlap_count: NonNegativeInt
    clean_raw_normalized_text_overlap_count: NonNegativeInt


class InventorySliceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slice_id: str
    domain: InventoryDomain
    scenario: str
    origin: ScenarioOrigin
    base_source_id: str
    status: SliceStatus
    transformation_ids: tuple[str, ...]
    artifact_path: str | None = None
    artifact_sha256: Sha256 | None = None
    parent_record_ids_sha256: Sha256 | None = None
    parent_record_count: PositiveInt | None = None
    materialized_record_count: PositiveInt | None = None
    unique_fixture_id_count: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_materialization_evidence(self) -> "InventorySliceEvidence":
        evidence = (self.artifact_path, self.artifact_sha256, self.parent_record_ids_sha256, self.parent_record_count, self.materialized_record_count, self.unique_fixture_id_count)
        if self.status is SliceStatus.MATERIALIZED:
            if any(value is None for value in evidence) or self.materialized_record_count != self.unique_fixture_id_count:
                raise DevelopmentCorpusInventoryError("materialized_slice_evidence_incomplete")
        elif any(value is not None for value in evidence):
            raise DevelopmentCorpusInventoryError("nonmaterialized_slice_claims_evidence")
        return self


class InventoryBuildEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    materialized_slices: tuple[InventorySliceEvidence, ...]
    cross_slice_parent_overlap_count: NonNegativeInt
    admission: InventoryAdmissionEvidence | None = None


class DevelopmentCorpusInventoryManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-corpus-inventory-manifest-v1"]
    status: InventoryStatus
    registry_sha256: Sha256
    sources: tuple[InventorySourceEvidence, ...]
    domain_pairs: tuple[DomainPairEvidence, ...]
    cross_source_record_id_overlap_count: NonNegativeInt
    cross_source_normalized_text_overlap_count: NonNegativeInt
    cross_slice_parent_overlap_count: NonNegativeInt | None
    slices: tuple[InventorySliceEvidence, ...]
    blocker_codes: tuple[str, ...]
    admission_report_sha256: Sha256 | None = None
    benchmark_exclusion_complete: bool = False
    frozen_confirmatory_domains: tuple[InventoryDomain, ...] = ()
    manifest_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    selector_membership_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_manifest(self) -> "DevelopmentCorpusInventoryManifest":
        source_pairs = {(item.domain, item.role) for item in self.sources}
        expected_source_pairs = {(domain, role) for domain in InventoryDomain for role in SourceRole}
        expected_slices = {
            f"{domain.value}-{scenario}"
            for domain in InventoryDomain
            for scenario in ("clean", "duplicate_heavy", "malformed", "boilerplate_heavy", "mixed_raw_like")
        }
        if source_pairs != expected_source_pairs or len(self.sources) != len(expected_source_pairs):
            raise DevelopmentCorpusInventoryError("inventory_manifest_source_matrix_incomplete")
        if {item.slice_id for item in self.slices} != expected_slices or len(self.slices) != len(expected_slices):
            raise DevelopmentCorpusInventoryError("inventory_manifest_scenario_matrix_incomplete")
        if {item.domain for item in self.domain_pairs} != set(InventoryDomain) or len(self.domain_pairs) != len(InventoryDomain):
            raise DevelopmentCorpusInventoryError("inventory_manifest_domain_pair_evidence_incomplete")
        admission_complete = self.benchmark_exclusion_complete and set(self.frozen_confirmatory_domains) == set(InventoryDomain)
        admitted = not self.blocker_codes and admission_complete and self.admission_report_sha256 is not None and self.cross_slice_parent_overlap_count == 0 and all(item.status is not SliceStatus.MATERIALIZATION_PENDING for item in self.slices)
        if (self.status is InventoryStatus.ADMITTED) != admitted:
            raise DevelopmentCorpusInventoryError("inventory_admission_status_evidence_mismatch")
        payload = {
            "registry_sha256": self.registry_sha256,
            "sources": [item.model_dump(mode="json") for item in self.sources],
            "domain_pairs": [item.model_dump(mode="json") for item in self.domain_pairs],
            "cross_source_record_id_overlap_count": self.cross_source_record_id_overlap_count,
            "cross_source_normalized_text_overlap_count": self.cross_source_normalized_text_overlap_count,
            "cross_slice_parent_overlap_count": self.cross_slice_parent_overlap_count,
            "slices": [item.model_dump(mode="json") for item in self.slices],
            "blocker_codes": list(self.blocker_codes),
            "admission_report_sha256": self.admission_report_sha256,
            "benchmark_exclusion_complete": self.benchmark_exclusion_complete,
            "frozen_confirmatory_domains": [item.value for item in self.frozen_confirmatory_domains],
        }
        if self.manifest_sha256 != hash_json(payload):
            legacy_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"admission_report_sha256", "benchmark_exclusion_complete", "frozen_confirmatory_domains"}
            }
            legacy_state = self.admission_report_sha256 is None and not self.benchmark_exclusion_complete and not self.frozen_confirmatory_domains
            if not legacy_state or self.manifest_sha256 != hash_json(legacy_payload):
                raise DevelopmentCorpusInventoryError("inventory_manifest_hash_mismatch")
        return self


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_inventory_registry(path: Path) -> DevelopmentCorpusInventoryRegistry:
    return DevelopmentCorpusInventoryRegistry.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "ConfirmatoryReference", "DevelopmentCorpusInventoryError", "DevelopmentCorpusInventoryManifest",
    "DevelopmentCorpusInventoryRegistry", "DomainPairEvidence", "InventoryDomain",
    "InventoryAdmissionEvidence", "InventoryBuildEvidence", "InventorySliceEvidence", "InventorySourceEvidence", "InventorySourceSpec", "InventoryStatus",
    "ScenarioOrigin", "SliceStatus", "SourceRole", "hash_json",
    "load_inventory_registry",
]
