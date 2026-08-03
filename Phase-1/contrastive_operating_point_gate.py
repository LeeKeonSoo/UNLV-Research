from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from contrastive_quality_contract import (
    CalibrationStatus,
    ModelRole,
    Precision,
    RoleQualification,
    load_contrastive_protocol,
)

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
DEFAULT_CONFIG = "configs/contrastive_operating_point_gate_v1.json"


@dataclass(frozen=True, slots=True)
class ContrastiveOperatingPointError(ValueError):
    reason_code: str
    path: str

    def __str__(self) -> str:
        return f"{self.reason_code}:{self.path}"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: Sha256


class CommonBaselineSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_sha256: Sha256
    record_ids_sha256: Sha256
    source_group_ids: tuple[str, ...] = Field(min_length=1)


class SensitivityArmSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: str = Field(min_length=1)
    route: str = Field(min_length=1)
    effect_bin_rank: int = Field(ge=1)
    artifact_sha256: Sha256
    record_ids_sha256: Sha256
    source_group_ids: tuple[str, ...] = Field(min_length=1)
    common_baseline_sha256: Sha256
    baseline_record_overlap_count: int = Field(ge=0)
    baseline_source_overlap_count: int = Field(ge=0)


class AcceptanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    require_qualified_three_role_provider: Literal[True]
    require_validated_execution_precision: Literal[True]
    require_one_shared_stage_a_baseline: Literal[True]
    require_baseline_disjoint_from_every_arm: Literal[True]
    require_arms_pairwise_disjoint: Literal[True]
    require_ordered_route_effect_bins: Literal[True]
    require_external_natural_budget_evidence: Literal[True]
    require_hard_subset_of_normal: Literal[True]


class SelectorBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_outcomes_available_at_runtime: Literal[False]
    utility_available_at_runtime: Literal[False]
    source_reputation_available_at_runtime: Literal[False]
    runtime_activation_mutation_allowed: Literal[False]


class GateProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["contrastive-operating-point-gate-protocol-v1"]
    status: Literal["block_10b_frozen_preflight"]
    contrastive_protocol: EvidenceRef
    contrastive_audit: EvidenceRef
    required_routes: tuple[str, ...] = Field(min_length=1)
    minimum_source_groups_per_route: int = Field(ge=3)
    minimum_ordered_effect_bins_per_route: int = Field(ge=3)
    profile_ids: tuple[Literal["normal", "hard"], ...]
    common_baseline: CommonBaselineSpec | None
    sensitivity_arms: tuple[SensitivityArmSpec, ...]
    arm_pairwise_disjointness_artifact_sha256: Sha256 | None
    external_natural_budget_evidence_sha256: Sha256 | None
    operating_point_artifact_sha256_by_profile: dict[Literal["normal", "hard"], Sha256 | None]
    profile_monotonicity_artifact_sha256: Sha256 | None
    acceptance: AcceptanceSpec
    selector_boundary: SelectorBoundary
    claim_boundary: str = Field(min_length=1)


class RouteAudit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    route: str
    source_group_count: int = Field(ge=0)


class FrozenAudit(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["blocked", "ready_for_effect_bin_experiment"]
    route_reports: tuple[RouteAudit, ...]
    blocker_codes: tuple[str, ...]
    benchmark_outcomes_read: Literal[False]
    utility_read: Literal[False]
    runtime_activation: Literal[False]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_path(root: Path, evidence: EvidenceRef) -> Path:
    path = root / evidence.path
    if _sha256(path) != evidence.sha256:
        raise ContrastiveOperatingPointError("contrastive_gate_evidence_hash_mismatch", evidence.path)
    return path


def _report_hash(report: JsonMap) -> str:
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_contrastive_operating_point_gate(root: Path) -> JsonMap:
    config_path = root / DEFAULT_CONFIG
    gate = GateProtocol.model_validate_json(config_path.read_text(encoding="utf-8"))
    protocol = load_contrastive_protocol(_verified_path(root, gate.contrastive_protocol))
    audit = FrozenAudit.model_validate_json(_verified_path(root, gate.contrastive_audit).read_text(encoding="utf-8"))
    by_role = {model.role: model for model in protocol.models}
    qualified_provider = (
        by_role[ModelRole.TARGET].role_qualification is RoleQualification.TARGET_SLM
        and by_role[ModelRole.QUALITY_REFERENCE].role_qualification
        is RoleQualification.VALIDATED_REFERENCE_POOL
        and by_role[ModelRole.BACKGROUND].role_qualification is RoleQualification.BROAD_BACKGROUND
    )
    precision_validated = all(
        model.precision not in {Precision.INT8, Precision.INT4}
        or model.quantization_validation_artifact_sha256 is not None
        for model in protocol.models
    )
    baseline = gate.common_baseline
    shared_baseline = baseline is not None and bool(gate.sensitivity_arms) and all(
        arm.common_baseline_sha256 == baseline.artifact_sha256 for arm in gate.sensitivity_arms
    )
    baseline_disjoint = shared_baseline and all(
        arm.baseline_record_overlap_count == 0 and arm.baseline_source_overlap_count == 0
        for arm in gate.sensitivity_arms
    )
    arms_pairwise_disjoint = (
        len(gate.sensitivity_arms) > 1 and gate.arm_pairwise_disjointness_artifact_sha256 is not None
    )
    route_audit = {item.route: item for item in audit.route_reports}
    route_gates: list[JsonValue] = []
    route_ready = True
    for route in gate.required_routes:
        source_count = route_audit[route].source_group_count if route in route_audit else 0
        effect_count = len({arm.effect_bin_rank for arm in gate.sensitivity_arms if arm.route == route})
        ready = (
            source_count >= gate.minimum_source_groups_per_route
            and effect_count >= gate.minimum_ordered_effect_bins_per_route
        )
        route_ready = route_ready and ready
        route_gates.append(
            {
                "route": route,
                "observed_source_group_count": source_count,
                "required_source_group_count": gate.minimum_source_groups_per_route,
                "observed_effect_bin_count": effect_count,
                "required_effect_bin_count": gate.minimum_ordered_effect_bins_per_route,
                "ready": ready,
            }
    )
    external_present = gate.external_natural_budget_evidence_sha256 is not None
    operating_points_present = set(gate.operating_point_artifact_sha256_by_profile) == set(
        gate.profile_ids
    ) and all(gate.operating_point_artifact_sha256_by_profile.values())
    monotonicity_verified = gate.profile_monotonicity_artifact_sha256 is not None
    ready = all(
        (
            protocol.calibration.status is CalibrationStatus.READY,
            qualified_provider,
            precision_validated,
            shared_baseline,
            baseline_disjoint,
            arms_pairwise_disjoint,
            route_ready,
            external_present,
            operating_points_present,
            monotonicity_verified,
        )
    )
    blockers = set(protocol.calibration.blocker_codes) | set(audit.blocker_codes)
    if not external_present:
        blockers.add("external_natural_budget_evidence_missing")
    if not operating_points_present:
        blockers.add("profile_operating_point_artifacts_missing")
    if not monotonicity_verified:
        blockers.add("profile_monotonicity_evidence_missing")
    report: JsonMap = {
        "schema_version": "contrastive-operating-point-gate-v1",
        "status": "ready" if ready else "blocked_missing_empirical_inputs",
        "gate_protocol_sha256": _sha256(config_path),
        "contrastive_protocol_sha256": gate.contrastive_protocol.sha256,
        "contrastive_audit_sha256": gate.contrastive_audit.sha256,
        "required_routes": list(gate.required_routes),
        "qualified_three_role_provider": qualified_provider,
        "validated_execution_precision": precision_validated,
        "common_baseline_shared_by_all_arms": shared_baseline,
        "baseline_disjoint_from_every_arm": baseline_disjoint,
        "arms_pairwise_disjoint": arms_pairwise_disjoint,
        "sensitivity_arm_count": len(gate.sensitivity_arms),
        "route_gates": route_gates,
        "external_natural_budget_evidence_present": external_present,
        "profile_operating_point_artifacts_present": operating_points_present,
        "hard_subset_of_normal_verified": monotonicity_verified,
        "operating_point_decisions": [
            {
                "profile_id": profile_id,
                "status": "ready" if ready else "blocked",
                "threshold_emitted": ready,
                "artifact_sha256": gate.operating_point_artifact_sha256_by_profile[profile_id],
            }
            for profile_id in gate.profile_ids
        ],
        "blocker_codes": [] if ready else sorted(blockers),
        "hard_subset_of_normal_required": True,
        "runtime_activation_mutated": False,
        "benchmark_outcomes_read_at_runtime": False,
        "utility_read_at_runtime": False,
        "claim_boundary": gate.claim_boundary,
    }
    report["report_sha256"] = _report_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Block 10B contrastive operating-point gate.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_contrastive_operating_point_gate(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blocker_count": len(report["blocker_codes"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
