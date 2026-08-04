from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TypeAlias

from coverage_contract import CoverageDecision
from joint_selector_contract import (
    JointProfileSpec,
    JointRemovalTrace,
    JointSelectionRequest,
    JointSelectionResult,
    JointSelectionStatus,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class JointResultParts:
    status: JointSelectionStatus
    reason_code: str
    selected_uids: tuple[str, ...]
    removal_traces: tuple[JointRemovalTrace, ...]
    coverage_decision: CoverageDecision | None
    evidence_artifact_hashes: tuple[str, ...]
    coverage_required_retain_uids: tuple[str, ...] = ()
    coverage_rematerialization_applied: bool = False


def _sha256(payload: dict[str, JsonValue]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _input_sha256(request: JointSelectionRequest) -> str:
    return _sha256(
        {
            "chunks": [(chunk.uid, chunk.token_count) for chunk in sorted(request.chunks, key=lambda item: item.uid)],
            "families": [
                (
                    family.family_id,
                    sorted(family.member_uids),
                    family.evidence_artifact_sha256,
                    family.preferred_representative_uid,
                )
                for family in sorted(request.redundancy_families, key=lambda item: item.family_id)
            ],
            "quality": [
                (
                    decision.chunk_uid,
                    decision.decision.value,
                    decision.reason_code,
                    decision.effect_direction.value if decision.effect_direction is not None else None,
                    decision.evidence_artifact_hashes,
                )
                for decision in sorted(request.quality_decisions, key=lambda item: item.chunk_uid)
            ],
            "strata": [
                (
                    stratum.stratum_id,
                    stratum.view.value,
                    sorted(stratum.member_uids),
                    stratum.state.value,
                    stratum.evidence_artifact_sha256,
                )
                for stratum in sorted(request.coverage_strata, key=lambda item: item.stratum_id)
            ],
            "similarities": [
                (edge.left_uid, edge.right_uid, edge.similarity, edge.evidence_artifact_sha256)
                for edge in sorted(request.similarities, key=lambda item: tuple(sorted((item.left_uid, item.right_uid))))
            ],
            "quality_provider": request.quality_provider_identity_sha256,
            "semantic_provider": request.semantic_provider_identity_sha256,
        }
    )


def finalize_joint_result(
    request: JointSelectionRequest,
    profile: JointProfileSpec,
    parts: JointResultParts,
) -> JointSelectionResult:
    input_sha = _input_sha256(request)
    profile_sha = profile.identity_sha256()
    policy_hashes = (profile_sha,)
    model_hashes = tuple(sorted({request.quality_provider_identity_sha256, request.semantic_provider_identity_sha256}))
    manifest_sha = _sha256(
        {
            "profile": profile_sha,
            "status": parts.status.value,
            "reason": parts.reason_code,
            "selected": parts.selected_uids,
            "removals": [asdict(trace) for trace in parts.removal_traces],
            "coverage": asdict(parts.coverage_decision) if parts.coverage_decision is not None else None,
            "coverage_required_retain_uids": parts.coverage_required_retain_uids,
            "coverage_rematerialization_applied": parts.coverage_rematerialization_applied,
            "input": input_sha,
            "policy_hashes": policy_hashes,
            "evidence_hashes": parts.evidence_artifact_hashes,
            "model_hashes": model_hashes,
        }
    )
    return JointSelectionResult(
        profile_name=profile.name,
        profile_sha256=profile_sha,
        status=parts.status,
        reason_code=parts.reason_code,
        selected_uids=parts.selected_uids,
        removal_traces=parts.removal_traces,
        coverage_decision=parts.coverage_decision,
        input_sha256=input_sha,
        manifest_sha256=manifest_sha,
        policy_artifact_hashes=policy_hashes,
        evidence_artifact_hashes=parts.evidence_artifact_hashes,
        model_provider_identity_hashes=model_hashes,
        coverage_required_retain_uids=parts.coverage_required_retain_uids,
        coverage_rematerialization_applied=parts.coverage_rematerialization_applied,
    )
