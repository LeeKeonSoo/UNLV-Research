from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
type JsonValue = str | int | float | bool | None | list[JsonValue] | tuple[JsonValue, ...] | dict[str, JsonValue]
Sha256 = Annotated[str, StringConstraints(pattern=SHA256_RE.pattern)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NORMAL_POLICY_IDS = (
    "candidate_redundancy_v2",
    "stage_c_calibrated_quality_effect_candidate",
    "stage_c_coverage_support_candidate",
)


def hash_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DevelopmentSelectionContractError(RuntimeError):
    reason_code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.reason_code}: {self.detail}"


class CorpusDomain(str, Enum):
    CODE = "code"
    MATH = "math"
    GENERAL = "general"


class CorpusScenario(str, Enum):
    CLEAN = "clean"
    DUPLICATE_HEAVY = "duplicate_heavy"
    MALFORMED = "malformed"
    BOILERPLATE_HEAVY = "boilerplate_heavy"
    MIXED_RAW_LIKE = "mixed_raw_like"


class ArmRole(str, Enum):
    NORMAL = "normal"
    HARD_EXTENSION = "hard_extension"


class DevelopmentSelectionStatus(str, Enum):
    FROZEN = "frozen"
    BLOCKED = "blocked"


class DevelopmentProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-selection-v1"]
    status: Literal["block_8_development_only_fail_closed"]
    required_domains: tuple[CorpusDomain, ...]
    required_scenarios: tuple[CorpusScenario, ...]
    one_sided_confidence_z: float
    maximum_clean_false_positive_upper_bound: float
    clean_control_bound_role: Literal["preregistered_engineering_risk_tolerance_not_universal_empirical_constant"]
    normal_selection_rule: Literal["maximize_gain_lcb_then_minimize_tokens_on_pareto_frontier"]
    hard_selection_rule: Literal["minimize_tokens_then_maximize_gain_lcb_on_pareto_frontier"]
    development_corpus_manifest: str
    hard_candidate_inventory: str
    hard_candidate_inventory_sha256: Sha256
    hard_extension_candidate_ids: tuple[str, ...]
    benchmark_outcomes_available_to_selector: Literal[False]
    confirmatory_outcomes_available_to_selector: Literal[False]
    source_reputation_available_to_selector: Literal[False]
    target_retention_fraction_allowed: Literal[False]
    hidden_token_budget_allowed: Literal[False]
    weighted_core_formula_allowed: Literal[False]
    post_run_override_allowed: Literal[False]

    @model_validator(mode="after")
    def validate_closed_matrix(self) -> "DevelopmentProtocol":
        if set(self.required_domains) != set(CorpusDomain) or len(self.required_domains) != len(CorpusDomain):
            raise DevelopmentSelectionContractError("development_domain_contract_drift", "Code, Math, and General are required exactly once")
        if set(self.required_scenarios) != set(CorpusScenario) or len(self.required_scenarios) != len(CorpusScenario):
            raise DevelopmentSelectionContractError("development_scenario_contract_drift", "All five scenarios are required exactly once")
        if self.one_sided_confidence_z <= 0 or not 0 < self.maximum_clean_false_positive_upper_bound < 1:
            raise DevelopmentSelectionContractError("development_statistical_boundary_invalid", "Confidence and clean-control bounds must be preregistered")
        if not self.hard_extension_candidate_ids or len(set(self.hard_extension_candidate_ids)) != len(self.hard_extension_candidate_ids):
            raise DevelopmentSelectionContractError("development_hard_inventory_invalid", "Hard candidates must be frozen and unique")
        return self

    def identity_sha256(self) -> str:
        return hash_json(self.model_dump(mode="json"))


class _HardCandidateInventory(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    selected_initial_candidates: tuple[str, ...]


class CorpusSliceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    slice_id: str
    domain: CorpusDomain
    scenario: CorpusScenario
    record_count: PositiveInt
    record_ids_artifact_sha256: Sha256
    normalized_text_hashes_artifact_sha256: Sha256
    confirmatory_record_overlap_count: NonNegativeInt
    confirmatory_text_overlap_count: NonNegativeInt
    confirmatory_source_snapshot_overlap_count: NonNegativeInt
    confirmatory_time_overlap_count: NonNegativeInt
    benchmark_exclusion_passed: bool
    evidence_artifact_sha256: Sha256


class DevelopmentGateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    redundancy_ready: bool
    quality_ready: bool
    coverage_ready: bool
    external_results_hidden: bool
    clean_control_count: PositiveInt
    clean_control_false_positives: NonNegativeInt
    provider_bias_stress_passed: bool
    route_holdout_stress_passed: bool
    evidence_artifact_hashes: tuple[Sha256, ...]


class DevelopmentArm(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    arm_id: str
    role: ArmRole
    required_policy_ids: tuple[str, ...]
    hard_extension_policy_ids: tuple[str, ...]
    exact_natural_tokens: PositiveInt
    development_gain_lcb_per_token: float
    effect_metric_id: str
    effect_metric_artifact_sha256: Sha256
    common_baseline_artifact_sha256: Sha256
    all_sensitivity_arms_share_one_common_baseline: bool
    common_baseline_disjoint_from_all_arms: bool
    development_and_heldout_effect_arms_disjoint: bool
    maximum_coverage_js_divergence: Annotated[float, Field(ge=0, le=1)]
    minimum_support_recall: Annotated[float, Field(ge=0, le=1)]
    extinct_supported_strata: NonNegativeInt
    unknown_mixed_extinct_strata: NonNegativeInt
    representative_linkage_complete: bool
    valid_residuals_complete: bool
    removal_trace_count: NonNegativeInt
    complete_removal_trace_count: NonNegativeInt
    input_manifest_sha256: Sha256
    evidence_artifact_hashes: tuple[Sha256, ...]


class DevelopmentSelectionBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: DevelopmentProtocol
    corpus_slices: tuple[CorpusSliceEvidence, ...]
    gates: DevelopmentGateEvidence
    base_exact_natural_tokens: PositiveInt
    base_input_manifest_sha256: Sha256
    arms: tuple[DevelopmentArm, ...]

    def validate_contract(self) -> None:
        expected = {(domain, scenario) for domain in self.protocol.required_domains for scenario in self.protocol.required_scenarios}
        observed = {(item.domain, item.scenario) for item in self.corpus_slices}
        if observed != expected or len(self.corpus_slices) != len(expected):
            raise DevelopmentSelectionContractError("development_matrix_incomplete", "Every required domain/scenario needs one evidence slice")
        if any(_slice_has_overlap(item) for item in self.corpus_slices):
            raise DevelopmentSelectionContractError("development_confirmatory_overlap", "Development and confirmatory evidence must be disjoint")
        if any(not item.benchmark_exclusion_passed for item in self.corpus_slices):
            raise DevelopmentSelectionContractError("development_benchmark_exclusion_failed", "Every admitted slice requires exclusion evidence")
        for arm in self.arms:
            _validate_arm(arm, self)
        effect_contracts = {(arm.effect_metric_id, arm.effect_metric_artifact_sha256, arm.common_baseline_artifact_sha256) for arm in self.arms}
        if len(effect_contracts) != 1:
            raise DevelopmentSelectionContractError("development_effect_contract_mismatch", "Every arm must share one metric and one common baseline")


class DevelopmentSelectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: DevelopmentSelectionStatus
    blocker_codes: tuple[str, ...]
    normal_arm_id: str | None
    hard_arm_id: str | None
    normal_profile_sha256: str | None
    hard_profile_sha256: str | None
    protocol_sha256: str
    manifest_sha256: str
    benchmark_outcomes_read: Literal[False] = False
    confirmatory_outcomes_read: Literal[False] = False
    target_retention_fraction_read: Literal[False] = False
    weighted_core_formula_used: Literal[False] = False


def _slice_has_overlap(item: CorpusSliceEvidence) -> bool:
    return any((item.confirmatory_record_overlap_count, item.confirmatory_text_overlap_count, item.confirmatory_source_snapshot_overlap_count, item.confirmatory_time_overlap_count))


def _validate_arm(arm: DevelopmentArm, bundle: DevelopmentSelectionBundle) -> None:
    if arm.required_policy_ids != NORMAL_POLICY_IDS or arm.input_manifest_sha256 != bundle.base_input_manifest_sha256:
        raise DevelopmentSelectionContractError("development_arm_identity_invalid", "Every arm must use the frozen Normal policies and Base input")
    match arm.role:
        case ArmRole.NORMAL:
            if arm.hard_extension_policy_ids:
                raise DevelopmentSelectionContractError("development_normal_extension_forbidden", "Normal cannot carry a Hard extension")
        case ArmRole.HARD_EXTENSION:
            if len(arm.hard_extension_policy_ids) != 1 or arm.hard_extension_policy_ids[0] not in bundle.protocol.hard_extension_candidate_ids:
                raise DevelopmentSelectionContractError("development_hard_extension_invalid", "Hard requires one registered extension")
        case unreachable:
            assert_never(unreachable)


def load_development_protocol(path: Path) -> DevelopmentProtocol:
    protocol = DevelopmentProtocol.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[1]
    inventory_path = (root / protocol.hard_candidate_inventory).resolve()
    if root not in inventory_path.parents or not inventory_path.is_file():
        raise DevelopmentSelectionContractError("development_hard_inventory_missing", "The frozen Hard inventory must remain inside the project root")
    if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != protocol.hard_candidate_inventory_sha256:
        raise DevelopmentSelectionContractError("development_hard_inventory_hash_mismatch", "The frozen Hard inventory hash changed")
    inventory = _HardCandidateInventory.model_validate_json(inventory_path.read_text(encoding="utf-8"))
    if inventory.selected_initial_candidates != protocol.hard_extension_candidate_ids:
        raise DevelopmentSelectionContractError("development_hard_inventory_candidate_mismatch", "Hard candidate IDs must match the frozen inventory")
    return protocol
