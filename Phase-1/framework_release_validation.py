from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from framework_equivalence_fixture import EquivalenceRecord, curated_projection_hash
from framework_objects import CoreId, ProviderSpec, StageId, ThresholdProvenance
from framework_profiles import (
    ProfileContractError,
    ProfileRegistry,
    validate_retained_set_monotonicity,
)
from framework_runtime_bridge import (
    RuntimeBridgeError,
    RuntimeStageRequest,
    authorize_runtime_stage,
    load_runtime_foundation,
)
from stage_permissions import StagePermissionError
from validation.core_behavior_audit_v3 import build_audit

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
ResultT = TypeVar("ResultT")


class ReleaseValidationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-release-validation-protocol-v1"]
    status: Literal["block_8_frozen_integrity_protocol"]
    core_fixture_path: str
    case_matrix_path: str
    policy_registry_path: str
    required_fixture_kinds: tuple[str, ...] = Field(min_length=1)
    blocked_policy_ids: tuple[str, ...] = Field(min_length=1)
    equivalence_records: tuple[EquivalenceRecord, ...] = Field(min_length=1)
    expected_curated_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_boundary: str = Field(min_length=1)


class IntegrityGate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    passed: bool
    expected_reason_code: str | None = None
    observed_reason_code: str | None = None
    expected_sha256: str | None = None
    observed_sha256: str | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expect_reason(gate_id: str, expected: str, operation: Callable[[], ResultT]) -> IntegrityGate:
    try:
        operation()
    except (RuntimeBridgeError, StagePermissionError, ProfileContractError) as error:
        observed = str(error)
        return IntegrityGate(
            id=gate_id,
            passed=observed == expected,
            expected_reason_code=expected,
            observed_reason_code=observed,
        )
    return IntegrityGate(id=gate_id, passed=False, expected_reason_code=expected)


def _expect_validation_error(gate_id: str, operation: Callable[[], ResultT]) -> IntegrityGate:
    try:
        operation()
    except ValidationError:
        code = f"{gate_id}_rejected"
        return IntegrityGate(
            id=gate_id,
            passed=True,
            expected_reason_code=code,
            observed_reason_code=code,
        )
    return IntegrityGate(id=gate_id, passed=False, expected_reason_code=f"{gate_id}_rejected")


def _tamper_kernel(root: Path) -> None:
    with TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        bridge = json.loads((root / "configs/framework_runtime_bridge_v1.json").read_text(encoding="utf-8"))
        for relative in (
            "configs/framework_runtime_bridge_v1.json",
            str(bridge["framework_manifest_path"]),
            str(bridge["object_registry_path"]),
            str(bridge["profile_registry_path"]),
            str(bridge["legacy_kernel_path"]),
        ):
            destination = temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, destination)
        kernel = temporary_root / str(bridge["legacy_kernel_path"])
        kernel.write_bytes(kernel.read_bytes() + b"\n# tampered\n")
        load_runtime_foundation(temporary_root)


def _integrity_gates(root: Path, protocol: ReleaseValidationProtocol) -> tuple[IntegrityGate, ...]:
    foundation = load_runtime_foundation(root)
    objects = json.loads((root / "configs/framework_objects_v1.json").read_text(encoding="utf-8"))
    profiles = json.loads((root / "configs/framework_profiles_v1.json").read_text(encoding="utf-8"))
    missing_provenance: dict[str, JsonValue] = {
        "value": 0.5,
        "unit": "fixture_unit",
        "comparison_direction": "greater_is_stronger_evidence",
        "derivation_procedure": "block_8_schema_fixture_only",
        "development_corpus_sha256": "0" * 64,
        "sample_count": 1,
        "supported_routes": ["fixture"],
        "provider_identity_sha256": "1" * 64,
        "tokenizer_identity_sha256": "2" * 64,
        "uncertainty_procedure": "fixture_only",
        "fixture_artifact_sha256": "3" * 64,
        "ablation_artifact_sha256": "4" * 64,
        "external_evidence_sha256": "5" * 64,
        "lifecycle": "candidate",
        "invalidation_conditions": ["fixture_only"],
    }
    missing_provenance.pop("external_evidence_sha256")
    provider = dict(objects["providers"][0])
    provider["direct_deletion_authority"] = True
    release_profiles = json.loads(json.dumps(profiles))
    release_profiles["profiles"][0]["release_enabled"] = True
    observed_hash = curated_projection_hash(protocol.equivalence_records)
    expected_hash = protocol.expected_curated_projection_sha256
    return (
        IntegrityGate(id="foundation_hash_chain", passed=foundation.schema_version == "framework-runtime-foundation-v1"),
        _expect_reason(
            "kernel_hash_tamper_detection",
            "runtime_bridge_kernel_identity_mismatch",
            lambda: _tamper_kernel(root),
        ),
        _expect_validation_error(
            "threshold_provenance_completeness",
            lambda: ThresholdProvenance.model_validate(missing_provenance),
        ),
        _expect_reason(
            "stage_core_authority",
            "stage_core_authority_mismatch:stage_b",
            lambda: authorize_runtime_stage(
                foundation,
                RuntimeStageRequest(
                    stage_id=StageId.STAGE_B,
                    core_id=CoreId.VALIDITY,
                    supplied_categories=("stage_a_survivors",),
                ),
            ),
        ),
        _expect_reason(
            "runtime_forbidden_input",
            "stage_runtime_forbidden_input:stage_b:benchmark_outcomes",
            lambda: authorize_runtime_stage(
                foundation,
                RuntimeStageRequest(
                    stage_id=StageId.STAGE_B,
                    core_id=CoreId.QUALITY,
                    supplied_categories=("benchmark_outcomes",),
                ),
            ),
        ),
        _expect_validation_error("provider_no_direct_deletion", lambda: ProviderSpec.model_validate(provider)),
        _expect_validation_error(
            "profile_no_uncalibrated_or_unpromoted_release",
            lambda: ProfileRegistry.model_validate(release_profiles),
        ),
        _expect_reason(
            "hard_retained_set_monotonicity",
            "profile_hard_retained_set_not_subset:hard-only",
            lambda: validate_retained_set_monotonicity(("normal",), ("normal", "hard-only")),
        ),
        IntegrityGate(
            id="curated_output_equivalence",
            passed=observed_hash == expected_hash,
            expected_sha256=expected_hash,
            observed_sha256=observed_hash,
        ),
    )


def build_release_validation(root: Path) -> JsonMap:
    protocol_path = root / "configs/framework_release_validation_v1.json"
    protocol = ReleaseValidationProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    core_audit = build_audit(root / protocol.core_fixture_path)
    fixture_kinds = sorted({str(case["fixture_kind"]) for case in core_audit["cases"]})
    core_passed = bool(core_audit["summary"]["core_behavior_gates_passed"]) and set(
        protocol.required_fixture_kinds
    ) <= set(fixture_kinds)
    gates = _integrity_gates(root, protocol)
    foundation = load_runtime_foundation(root)
    lifecycle_by_id = {policy.id: policy.lifecycle.value for policy in foundation.objects.policies}
    release_blockers = list(foundation.profiles.blocker_codes) + [
        f"{policy_id}:{lifecycle_by_id[policy_id]}"
        for policy_id in protocol.blocked_policy_ids
        if lifecycle_by_id[policy_id] != "promoted"
    ]
    integrity_passed = core_passed and all(gate.passed for gate in gates)
    report: JsonMap = {
        "schema_version": "framework-release-validation-v1",
        "protocol_sha256": _sha256(protocol_path),
        "implementation_integrity": "passed" if integrity_passed else "failed",
        "framework_release": "eligible" if integrity_passed and not release_blockers else "blocked",
        "core_behavior": {
            "passed": core_passed,
            "fixture_sha256": _sha256(root / protocol.core_fixture_path),
            "case_matrix_sha256": _sha256(root / protocol.case_matrix_path),
            "policy_registry_sha256": _sha256(root / protocol.policy_registry_path),
            "fixture_kinds": fixture_kinds,
            "summary": core_audit["summary"],
            "cores": core_audit["cores"],
        },
        "integrity_gates": [gate.model_dump(mode="json") for gate in gates],
        "release_blockers": release_blockers,
        "claim_boundary": protocol.claim_boundary,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Block 8 release-validation bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_release_validation(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"implementation_integrity": report["implementation_integrity"], "framework_release": report["framework_release"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
