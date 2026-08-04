from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from quality_teacher_panel import TeacherPanel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
EvidenceT = TypeVar("EvidenceT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class PolicyAblationError(ValueError):
    reason_code: str
    evidence_path: str

    def __str__(self) -> str:
        return f"{self.reason_code}:{self.evidence_path}"


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    development_admission: EvidenceRef
    redundancy_gate: EvidenceRef
    quality_teacher_panel: EvidenceRef


class PolicyAblationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["framework-policy-ablation-protocol-v1"]
    status: Literal["block_9_frozen_development_protocol"]
    evidence: EvidenceBundle
    exact_policy_id: Literal["redundancy.exact_text_family"]
    near_policy_id: Literal["redundancy.symmetric_near_duplicate_candidate"]
    quality_policy_id: Literal["quality.teacher_panel_candidate"]
    claim_boundary: str = Field(min_length=1)


class AdmissionEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["admitted"]
    benchmark_exclusion_complete: Literal[True]
    total_benchmark_contaminated_record_count: int = Field(ge=0)
    total_confirmatory_development_record_id_overlap_count: int = Field(ge=0)
    total_confirmatory_development_text_overlap_count: int = Field(ge=0)
    blocker_codes: tuple[str, ...]
    benchmark_outcomes_read: Literal[False]
    selector_membership_mutated: Literal[False]


class RedundancyEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    status: Literal["passed"]
    expected_exact_family_count: int = Field(ge=1)
    recovered_exact_family_count: int = Field(ge=0)
    expected_exact_copy_count: int = Field(ge=1)
    linked_exact_copy_count: int = Field(ge=1)
    clean_false_merged_record_count: int = Field(ge=0)
    perturbation_candidate_relation_count: int = Field(ge=0)
    cross_parent_safe_family_count: int = Field(ge=0)
    blocker_codes: tuple[str, ...]
    benchmark_outcomes_read: Literal[False]
    utility_read: Literal[False]
    selector_membership_mutated: Literal[False]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evidence(root: Path, reference: EvidenceRef, model: type[EvidenceT]) -> EvidenceT:
    path = root / reference.path
    observed = _sha256(path)
    if observed != reference.sha256:
        raise PolicyAblationError("block_9_evidence_hash_mismatch", reference.path)
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _report_hash(report: JsonMap) -> str:
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_policy_ablation(root: Path) -> JsonMap:
    protocol_path = root / "configs/framework_policy_ablation_v1.json"
    protocol = PolicyAblationProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    admission = _load_evidence(root, protocol.evidence.development_admission, AdmissionEvidence)
    redundancy = _load_evidence(root, protocol.evidence.redundancy_gate, RedundancyEvidence)
    teacher_panel = _load_evidence(root, protocol.evidence.quality_teacher_panel, TeacherPanel)
    admission_passed = (
        admission.benchmark_exclusion_complete
        and not admission.blocker_codes
        and admission.total_benchmark_contaminated_record_count == 0
        and admission.total_confirmatory_development_record_id_overlap_count == 0
        and admission.total_confirmatory_development_text_overlap_count == 0
    )
    exact_passed = (
        admission_passed
        and not redundancy.blocker_codes
        and redundancy.recovered_exact_family_count == redundancy.expected_exact_family_count
        and redundancy.linked_exact_copy_count == redundancy.expected_exact_copy_count
        and redundancy.clean_false_merged_record_count == 0
        and redundancy.cross_parent_safe_family_count == 0
    )
    near_blockers = ["near_positive_nonexact_equivalence_missing", "near_threshold_not_identifiable"]
    quality_blockers = [
        "quality_teacher_panel_not_runtime_active",
        "quality_teacher_smoke_fixture_matrix_missing",
        "quality_teacher_protected_fixture_gate_missing",
        "quality_teacher_consensus_stability_missing",
        "quality_normal_operating_point_missing",
        "quality_hard_operating_point_missing",
    ]
    decisions: list[JsonValue] = [
        {
            "policy_id": protocol.exact_policy_id,
            "lifecycle_decision": "development_passed" if exact_passed else "blocked",
            "positive_units": redundancy.linked_exact_copy_count,
            "false_positive_units": redundancy.clean_false_merged_record_count,
            "representative_failures": (
                redundancy.expected_exact_family_count - redundancy.recovered_exact_family_count
            ),
            "threshold_emitted": False,
            "blocker_codes": [] if exact_passed else ["exact_family_development_gate_failed"],
        },
        {
            "policy_id": protocol.near_policy_id,
            "lifecycle_decision": "blocked",
            "candidate_units": redundancy.perturbation_candidate_relation_count,
            "threshold_emitted": False,
            "blocker_codes": near_blockers,
        },
        {
            "policy_id": protocol.quality_policy_id,
            "lifecycle_decision": "blocked",
            "teacher_count": len(teacher_panel.teachers),
            "policy_count": len(teacher_panel.policies),
            "smoke_fixture_target": teacher_panel.fixture_matrix.total,
            "protected_fixture_target": teacher_panel.promotion_gate.protected_fixture_count,
            "operating_point_emitted": False,
            "blocker_codes": quality_blockers,
        },
    ]
    report: JsonMap = {
        "schema_version": "framework-policy-ablation-v1",
        "status": "block_9_complete_no_hard_policy_promoted",
        "protocol_sha256": _sha256(protocol_path),
        "development_admission": {
            "passed": admission_passed,
            "benchmark_contaminated_records": admission.total_benchmark_contaminated_record_count,
            "confirmatory_record_overlap": admission.total_confirmatory_development_record_id_overlap_count,
            "confirmatory_text_overlap": admission.total_confirmatory_development_text_overlap_count,
        },
        "evidence_sha256": {
            name: reference.sha256 for name, reference in protocol.evidence
        },
        "policy_decisions": decisions,
        "hard_profile_development_ready": False,
        "block_10_authorized": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "runtime_activation_mutated": False,
        "claim_boundary": protocol.claim_boundary,
    }
    report["report_sha256"] = _report_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Block 9 policy-ablation decision bundle.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_policy_ablation(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "block_10_authorized": report["block_10_authorized"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
