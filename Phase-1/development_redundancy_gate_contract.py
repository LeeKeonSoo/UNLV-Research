from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from development_corpus_inventory_contract import InventoryDomain
from redundancy_v2 import RedundancySettings, RelationType


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
NonNegativeInt = Annotated[int, Field(ge=0)]
type JsonValue = str | int | float | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DevelopmentRedundancyGateError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class RedundancyGateStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class RedundancySettingsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    short_exact_only_max_tokens: int = Field(ge=1)
    near_min_tokens: int = Field(ge=2)
    near_max_changed_ratio: float = Field(gt=0.0, lt=1.0)
    near_max_changed_tokens: int = Field(ge=1)
    containment_min_tokens: int = Field(ge=2)
    repeated_span_min_lexical_tokens: int = Field(ge=2)
    complementary_overlap_floor: float = Field(ge=0.0, le=1.0)

    def to_settings(self) -> RedundancySettings:
        return RedundancySettings(**self.model_dump())


class DevelopmentRedundancyGateRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-redundancy-gate-registry-v1"]
    status: Literal["e2-frozen-redundancy-inputs"]
    inventory_manifest_path: str
    inventory_manifest_sha256: Sha256
    inventory_manifest_file_sha256: Sha256
    required_domains: tuple[InventoryDomain, ...]
    required_scenarios: tuple[str, ...]
    safe_family_relations: tuple[RelationType, ...]
    candidate_only_relations: tuple[RelationType, ...]
    parent_relation: str
    exact_copy_relations: tuple[str, ...]
    perturbation_relation: str
    upstream_owned_relations: tuple[str, ...]
    settings: RedundancySettingsSpec
    confidence_level: float = Field(gt=0.5, lt=1.0)
    maximum_clean_false_merge_upper_bound: float = Field(gt=0.0, lt=1.0)
    maximum_perturbation_safe_merge_upper_bound: float = Field(gt=0.0, lt=1.0)
    benchmark_outcomes_available: Literal[False]
    utility_available: Literal[False]
    selector_membership_mutation_allowed: Literal[False]

    @model_validator(mode="after")
    def validate_closed_registry(self) -> "DevelopmentRedundancyGateRegistry":
        expected_scenarios = {"clean", "duplicate_heavy", "malformed", "boilerplate_heavy", "mixed_raw_like"}
        if set(self.required_domains) != set(InventoryDomain) or len(self.required_domains) != len(InventoryDomain):
            raise DevelopmentRedundancyGateError("redundancy_gate_domain_matrix_incomplete")
        if set(self.required_scenarios) != expected_scenarios or len(self.required_scenarios) != len(expected_scenarios):
            raise DevelopmentRedundancyGateError("redundancy_gate_scenario_matrix_incomplete")
        if set(self.safe_family_relations) != {RelationType.EXACT_EQUIVALENT, RelationType.FORMATTING_EQUIVALENT}:
            raise DevelopmentRedundancyGateError("redundancy_gate_safe_relations_invalid")
        if set(self.safe_family_relations) & set(self.candidate_only_relations):
            raise DevelopmentRedundancyGateError("redundancy_gate_relation_authority_overlap")
        if len(set(self.exact_copy_relations)) != len(self.exact_copy_relations):
            raise DevelopmentRedundancyGateError("redundancy_gate_exact_labels_not_unique")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class DevelopmentFixtureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fixture_id: str
    slice_id: str
    parent_record_id: str
    metamorphic_relation: str
    text: str
    normalized_text_sha256: Sha256

    @model_validator(mode="after")
    def validate_normalized_hash(self) -> "DevelopmentFixtureRecord":
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", self.text)).strip()
        if hashlib.sha256(normalized.encode()).hexdigest() != self.normalized_text_sha256:
            raise DevelopmentRedundancyGateError("redundancy_fixture_normalized_hash_mismatch")
        return self


@dataclass(frozen=True, slots=True)
class RedundancySliceInput:
    path: Path
    slice_id: str
    domain: InventoryDomain
    scenario: str
    expected_sha256: str


class RelationCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relation: RelationType
    count: NonNegativeInt


class SliceRedundancyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slice_id: str
    domain: InventoryDomain
    scenario: str
    artifact_sha256: Sha256
    record_count: NonNegativeInt
    evaluated_record_count: NonNegativeInt
    upstream_owned_record_count: NonNegativeInt
    safe_family_count: NonNegativeInt
    safe_family_member_count: NonNegativeInt
    cross_parent_safe_family_count: NonNegativeInt
    expected_exact_family_count: NonNegativeInt
    recovered_exact_family_count: NonNegativeInt
    expected_exact_copy_count: NonNegativeInt
    linked_exact_copy_count: NonNegativeInt
    perturbation_record_count: NonNegativeInt
    perturbation_safe_merge_count: NonNegativeInt
    perturbation_candidate_relation_count: NonNegativeInt
    clean_false_merged_record_count: NonNegativeInt
    relation_counts: tuple[RelationCount, ...]


class DevelopmentRedundancyGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-redundancy-gate-report-v1"]
    status: RedundancyGateStatus
    registry_sha256: Sha256
    inventory_manifest_sha256: Sha256
    inventory_manifest_file_sha256: Sha256
    slices: tuple[SliceRedundancyEvidence, ...]
    matrix_complete: bool
    expected_exact_family_count: NonNegativeInt
    recovered_exact_family_count: NonNegativeInt
    expected_exact_copy_count: NonNegativeInt
    linked_exact_copy_count: NonNegativeInt
    clean_control_record_count: NonNegativeInt
    clean_false_merged_record_count: NonNegativeInt
    clean_false_merge_upper_bound: float = Field(ge=0.0, le=1.0)
    perturbation_record_count: NonNegativeInt
    perturbation_safe_merge_count: NonNegativeInt
    perturbation_safe_merge_upper_bound: float = Field(ge=0.0, le=1.0)
    perturbation_candidate_relation_count: NonNegativeInt
    cross_parent_safe_family_count: NonNegativeInt
    blocker_codes: tuple[str, ...]
    report_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    utility_read: Literal[False] = False
    selector_membership_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "DevelopmentRedundancyGateReport":
        if (self.status is RedundancyGateStatus.PASSED) != (not self.blocker_codes):
            raise DevelopmentRedundancyGateError("redundancy_gate_status_evidence_mismatch")
        payload = self.model_dump(mode="json", exclude={"status", "report_sha256"})
        if self.report_sha256 != hash_json(payload):
            raise DevelopmentRedundancyGateError("redundancy_gate_report_hash_mismatch")
        return self


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_redundancy_gate_registry(path: Path) -> DevelopmentRedundancyGateRegistry:
    return DevelopmentRedundancyGateRegistry.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "DevelopmentFixtureRecord", "DevelopmentRedundancyGateError", "DevelopmentRedundancyGateRegistry",
    "DevelopmentRedundancyGateReport", "RedundancyGateStatus", "RedundancySettingsSpec",
    "RedundancySliceInput", "RelationCount", "SliceRedundancyEvidence", "hash_json",
    "load_redundancy_gate_registry",
]
