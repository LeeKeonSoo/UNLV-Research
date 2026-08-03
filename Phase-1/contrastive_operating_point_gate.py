from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from contrastive_quality_contract import (
    CalibrationStatus,
    ModelRole,
    Precision,
    RoleQualification,
    load_contrastive_protocol,
)
from contrastive_operating_point_contract import (
    EffectBinManifest,
    EvidenceRef,
    FrozenAudit,
    GateProtocol,
)

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
    arm_pool_aligned = (
        {arm.profile_id for arm in gate.sensitivity_arms} == set(gate.profile_ids)
        and len({arm.eligible_record_ids_sha256 for arm in gate.sensitivity_arms}) == 1
    )
    effect_manifest = (
        EffectBinManifest.model_validate_json(
            _verified_path(root, gate.effect_bin_manifest).read_text(encoding="utf-8")
        )
        if gate.effect_bin_manifest is not None
        else None
    )
    eligible_pool_sha256 = (
        gate.sensitivity_arms[0].eligible_record_ids_sha256 if arm_pool_aligned else None
    )
    effect_manifest_aligned = (
        effect_manifest is not None
        and baseline is not None
        and effect_manifest.common_baseline_sha256 == baseline.artifact_sha256
        and effect_manifest.eligible_record_ids_sha256 == eligible_pool_sha256
    )
    route_audit = {item.route: item for item in audit.route_reports}
    route_gates: list[JsonValue] = []
    route_ready = True
    for route in gate.required_routes:
        source_count = route_audit[route].source_group_count if route in route_audit else 0
        effect_count = (
            len({item.rank for item in effect_manifest.bins if item.route == route})
            if effect_manifest_aligned and effect_manifest is not None
            else 0
        )
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
            arm_pool_aligned,
            effect_manifest_aligned,
            route_ready,
            external_present,
            operating_points_present,
            monotonicity_verified,
        )
    )
    blockers = set(protocol.calibration.blocker_codes) | set(audit.blocker_codes)
    if not external_present:
        blockers.add("external_natural_budget_evidence_missing")
    if not arm_pool_aligned:
        blockers.add("sensitivity_arm_pool_missing")
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
        "sensitivity_arms_share_eligible_pool": arm_pool_aligned,
        "effect_bin_manifest_aligned": effect_manifest_aligned,
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
