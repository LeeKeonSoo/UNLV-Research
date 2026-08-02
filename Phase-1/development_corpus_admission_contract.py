from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from development_corpus_inventory_contract import InventoryDomain


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
type JsonValue = str | int | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DevelopmentCorpusAdmissionError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class AdmissionStatus(str, Enum):
    ADMITTED = "admitted"
    BLOCKED = "blocked"


class CorpusRole(str, Enum):
    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"


class BenchmarkArtifactFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"


class FilterLineageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_path: str
    source_sha256: Sha256
    evidence_path: str
    evidence_sha256: Sha256
    removed_record_count: PositiveInt


class CorpusReferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_id: str
    domain: InventoryDomain
    source_group_id: str
    source_snapshot_id: str
    path: str
    id_fields: tuple[str, ...]
    text_field: str
    expected_file_sha256: Sha256
    expected_record_count: PositiveInt
    selector_visible_source_metadata: Literal[False]
    filter_lineage: FilterLineageSpec | None = None


class BenchmarkArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str
    domain: InventoryDomain
    path: str
    expected_file_sha256: Sha256
    artifact_format: BenchmarkArtifactFormat


class DevelopmentCorpusAdmissionRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-corpus-admission-registry-v1"]
    status: Literal["e1-frozen-admission-inputs"]
    normalization: Literal["unicode-nfkc-casefold-whitespace-token-v1"]
    minimum_exact_segment_lexical_tokens: Annotated[int, Field(ge=8)]
    minimum_containment_segment_lexical_tokens: Annotated[int, Field(ge=13)]
    development_sources: tuple[CorpusReferenceSpec, ...]
    confirmatory_references: tuple[CorpusReferenceSpec, ...]
    benchmark_artifacts: tuple[BenchmarkArtifactSpec, ...]
    benchmark_outcomes_available: Literal[False]
    selector_membership_mutation_allowed: Literal[False]

    @model_validator(mode="after")
    def validate_closed_registry(self) -> "DevelopmentCorpusAdmissionRegistry":
        if {item.domain for item in self.development_sources} != set(InventoryDomain):
            raise DevelopmentCorpusAdmissionError("development_sources_incomplete")
        if len({item.reference_id for item in self.development_sources}) != len(self.development_sources):
            raise DevelopmentCorpusAdmissionError("development_source_ids_not_unique")
        if {item.domain for item in self.confirmatory_references} != set(InventoryDomain):
            raise DevelopmentCorpusAdmissionError("confirmatory_references_incomplete")
        if len(self.confirmatory_references) != len(InventoryDomain):
            raise DevelopmentCorpusAdmissionError("confirmatory_reference_count_invalid")
        if {item.domain for item in self.benchmark_artifacts} != set(InventoryDomain):
            raise DevelopmentCorpusAdmissionError("benchmark_domain_coverage_incomplete")
        if len({item.benchmark_id for item in self.benchmark_artifacts}) != len(self.benchmark_artifacts):
            raise DevelopmentCorpusAdmissionError("benchmark_artifact_ids_not_unique")
        development_groups = {item.source_group_id for item in self.development_sources}
        confirmatory_groups = {item.source_group_id for item in self.confirmatory_references}
        if development_groups & confirmatory_groups:
            raise DevelopmentCorpusAdmissionError("development_confirmatory_source_groups_overlap")
        if self.minimum_containment_segment_lexical_tokens < self.minimum_exact_segment_lexical_tokens:
            raise DevelopmentCorpusAdmissionError("containment_evidence_too_short")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class BenchmarkArtifactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    benchmark_id: str
    domain: InventoryDomain
    file_sha256: Sha256
    eligible_segment_count: PositiveInt


class CorpusBenchmarkScanEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_id: str
    domain: InventoryDomain
    role: CorpusRole
    record_count: PositiveInt
    exact_text_match_count: NonNegativeInt
    segment_containment_match_count: NonNegativeInt
    contaminated_record_count: NonNegativeInt


class ConfirmatoryDisjointEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: InventoryDomain
    reference_id: str
    source_group_id: str
    source_snapshot_id: str
    file_sha256: Sha256
    record_count: PositiveInt
    development_record_id_overlap_count: NonNegativeInt
    development_normalized_text_overlap_count: NonNegativeInt


class BenchmarkContaminationMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference_id: str
    domain: InventoryDomain
    role: CorpusRole
    record_id: str
    benchmark_id: str
    match_kind: Literal["exact_text", "segment_containment"]
    segment_sha256: Sha256
    segment_lexical_token_count: PositiveInt


class FilteredConfirmatoryReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["filtered-confirmatory-reference-v1"]
    reference_id: str
    domain: InventoryDomain
    source_path: str
    source_sha256: Sha256
    output_path: str
    output_sha256: Sha256
    input_record_count: PositiveInt
    output_record_count: PositiveInt
    removed_record_ids: tuple[str, ...]
    admission_report_sha256: Sha256


class DevelopmentCorpusAdmissionReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-corpus-admission-report-v1"]
    status: AdmissionStatus
    registry_sha256: Sha256
    benchmark_artifacts: tuple[BenchmarkArtifactEvidence, ...]
    corpus_scans: tuple[CorpusBenchmarkScanEvidence, ...]
    confirmatory_references: tuple[ConfirmatoryDisjointEvidence, ...]
    contamination_matches: tuple[BenchmarkContaminationMatch, ...]
    benchmark_exclusion_complete: bool
    frozen_confirmatory_domains: tuple[InventoryDomain, ...]
    total_benchmark_contaminated_record_count: NonNegativeInt
    total_confirmatory_development_record_id_overlap_count: NonNegativeInt
    total_confirmatory_development_text_overlap_count: NonNegativeInt
    blocker_codes: tuple[str, ...]
    report_sha256: Sha256
    benchmark_outcomes_read: Literal[False] = False
    selector_membership_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "DevelopmentCorpusAdmissionReport":
        admitted = not self.blocker_codes and self.benchmark_exclusion_complete
        if (self.status is AdmissionStatus.ADMITTED) != admitted:
            raise DevelopmentCorpusAdmissionError("admission_status_evidence_mismatch")
        payload = self.model_dump(mode="json", exclude={"report_sha256", "status"})
        if self.report_sha256 != hash_json(payload):
            raise DevelopmentCorpusAdmissionError("admission_report_hash_mismatch")
        return self


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_admission_registry(path: Path) -> DevelopmentCorpusAdmissionRegistry:
    return DevelopmentCorpusAdmissionRegistry.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "AdmissionStatus", "BenchmarkArtifactEvidence", "BenchmarkArtifactFormat", "BenchmarkContaminationMatch",
    "BenchmarkArtifactSpec", "ConfirmatoryDisjointEvidence", "CorpusBenchmarkScanEvidence",
    "CorpusReferenceSpec", "CorpusRole", "DevelopmentCorpusAdmissionRegistry", "FilterLineageSpec",
    "DevelopmentCorpusAdmissionError", "DevelopmentCorpusAdmissionReport", "FilteredConfirmatoryReference", "hash_json", "load_admission_registry",
]
