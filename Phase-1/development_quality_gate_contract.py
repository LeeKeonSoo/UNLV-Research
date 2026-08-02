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
NonNegativeInt = Annotated[int, Field(ge=0)]
type JsonValue = str | int | float | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DevelopmentQualityGateError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class QualityGateStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"


class QualityRouteRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: InventoryDomain
    route: str


class EmpiricalEffectBundleReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: InventoryDomain
    route: str
    artifact_path: str
    artifact_file_sha256: Sha256


class DevelopmentQualityGateRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-quality-gate-registry-v1"]
    status: Literal["e3-frozen-quality-inputs"]
    inventory_manifest_path: str
    inventory_manifest_sha256: Sha256
    inventory_manifest_file_sha256: Sha256
    route_evidence_gate_path: str
    route_evidence_gate_file_sha256: Sha256
    provider_registry_path: str
    provider_registry_file_sha256: Sha256
    contract_fixture_path: str
    contract_fixture_file_sha256: Sha256
    required_domains: tuple[InventoryDomain, ...]
    required_routes: tuple[QualityRouteRequirement, ...]
    empirical_effect_bundles: tuple[EmpiricalEffectBundleReference, ...]
    minimum_ordered_bins_per_route: int = Field(ge=3)
    common_baseline_artifact_sha256: Sha256 | None
    contract_fixture_may_satisfy_empirical_gate: Literal[False]
    benchmark_outcomes_available: Literal[False]
    utility_available: Literal[False]
    selector_membership_mutation_allowed: Literal[False]

    @model_validator(mode="after")
    def validate_closed_registry(self) -> "DevelopmentQualityGateRegistry":
        if set(self.required_domains) != set(InventoryDomain) or len(self.required_domains) != len(InventoryDomain):
            raise DevelopmentQualityGateError("quality_gate_domain_matrix_incomplete")
        route_domains = tuple(item.domain for item in self.required_routes)
        if set(route_domains) != set(self.required_domains) or len(route_domains) != len(self.required_domains):
            raise DevelopmentQualityGateError("quality_gate_route_matrix_incomplete")
        bundle_domains = tuple(item.domain for item in self.empirical_effect_bundles)
        if len(bundle_domains) != len(set(bundle_domains)):
            raise DevelopmentQualityGateError("quality_gate_duplicate_effect_bundle")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class EmpiricalEffectInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    point: float
    lower: float
    upper: float
    samples: int = Field(ge=3)


class EmpiricalEffectBin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bin_id: str
    bin_order: NonNegativeInt
    development: EmpiricalEffectInterval
    heldout: EmpiricalEffectInterval
    artifact_sha256: Sha256


class EmpiricalQualityEffectBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["empirical-quality-effect-bundle-v1"]
    contract_fixture_only: Literal[False]
    domain: InventoryDomain
    route: str
    provider_id: str
    provider_identity_sha256: Sha256
    effect_metric_id: str
    effect_metric_artifact_sha256: Sha256
    common_baseline_artifact_sha256: Sha256
    bins: tuple[EmpiricalEffectBin, ...]
    provider_training_source_groups: tuple[str, ...]
    development_source_groups: tuple[str, ...]
    heldout_source_groups: tuple[str, ...]
    all_arms_share_common_baseline: bool
    common_baseline_disjoint_from_all_bins: bool
    external_results_hidden: bool
    provider_bias_stress_passed: bool
    route_holdout_stress_passed: bool


class QualityRouteEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    domain: InventoryDomain
    route: str
    route_gate_decision: str
    route_evidence_ready: bool
    provider_id: str | None
    provider_lifecycle: str | None
    provider_policy_contribution_authority: bool
    effect_calibration_artifact: str | None
    empirical_effect_bin_count: NonNegativeInt
    calibration_passed: bool
    blocker_codes: tuple[str, ...]


class DevelopmentQualityGateReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-quality-gate-report-v1"]
    status: QualityGateStatus
    registry_sha256: Sha256
    inventory_manifest_sha256: Sha256
    inventory_manifest_file_sha256: Sha256
    routes: tuple[QualityRouteEvidence, ...]
    matrix_complete: bool
    contract_fixture_excluded: bool
    provider_active: bool
    empirical_effect_calibration_complete: bool
    common_baseline_empirically_verified: bool
    blocker_codes: tuple[str, ...]
    report_sha256: Sha256
    runtime_activation: Literal[False] = False
    benchmark_outcomes_read: Literal[False] = False
    utility_read: Literal[False] = False
    selector_membership_mutated: Literal[False] = False

    @model_validator(mode="after")
    def validate_report(self) -> "DevelopmentQualityGateReport":
        if (self.status is QualityGateStatus.PASSED) != (not self.blocker_codes):
            raise DevelopmentQualityGateError("quality_gate_status_evidence_mismatch")
        payload = self.model_dump(mode="json", exclude={"status", "report_sha256"})
        if self.report_sha256 != hash_json(payload):
            raise DevelopmentQualityGateError("quality_gate_report_hash_mismatch")
        return self


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_quality_gate_registry(path: Path) -> DevelopmentQualityGateRegistry:
    return DevelopmentQualityGateRegistry.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = [
    "DevelopmentQualityGateError",
    "DevelopmentQualityGateRegistry",
    "DevelopmentQualityGateReport",
    "EmpiricalQualityEffectBundle",
    "QualityGateStatus",
    "QualityRouteEvidence",
    "hash_json",
    "load_quality_gate_registry",
]
