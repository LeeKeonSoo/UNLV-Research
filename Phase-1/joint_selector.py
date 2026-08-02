from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

from coverage_contract import (
    CoverageRequest,
    CoverageStatus,
    ExclusionEvidence,
    ExclusionKind,
)
from coverage_engine import evaluate_coverage
from joint_selector_contract import (
    JointGateBundle,
    JointGateOrigin,
    JointProfileName,
    JointProfileRegistry,
    JointProfileSpec,
    JointRemovalTrace,
    JointSelectionRequest,
    JointSelectionResult,
    JointSelectionStatus,
    JointSelectorContractError,
    load_joint_profiles,
)
from joint_selector_gates import load_current_joint_gates
from joint_selector_manifest import JointResultParts, finalize_joint_result
from model_provider_contract import ProviderManifest
from quality_effect_engine import QualityEffectDecisionName


@dataclass(frozen=True, slots=True)
class _BaseOutcome:
    status: JointSelectionStatus
    reason_code: str
    evidence_artifact_hashes: tuple[str, ...]


def _retain_base(
    request: JointSelectionRequest,
    profile: JointProfileSpec,
    outcome: _BaseOutcome,
) -> JointSelectionResult:
    parts = JointResultParts(
        outcome.status,
        outcome.reason_code,
        tuple(sorted(chunk.uid for chunk in request.chunks)),
        (),
        None,
        outcome.evidence_artifact_hashes,
    )
    return finalize_joint_result(request, profile, parts)


def _normal_candidate(
    request: JointSelectionRequest,
    profile: JointProfileSpec,
    gates: JointGateBundle,
    semantic_provider: ProviderManifest,
) -> JointSelectionResult:
    quality_by_uid = {decision.chunk_uid: decision for decision in request.quality_decisions}
    quality_rejected = frozenset(
        uid for uid, decision in quality_by_uid.items() if decision.decision is QualityEffectDecisionName.REJECT_CANDIDATE
    )
    family_members = frozenset(uid for family in request.redundancy_families for uid in family.member_uids)
    universe = frozenset(chunk.uid for chunk in request.chunks)
    proposed = universe - family_members - quality_rejected
    exclusions = tuple(
        ExclusionEvidence(
            chunk_uid=uid,
            kind=ExclusionKind.QUALITY_SUPPORTED_NONPOSITIVE,
            policy_id="stage_c_calibrated_quality_effect_candidate",
            reason_code=quality_by_uid[uid].reason_code,
            evidence_artifact_hashes=quality_by_uid[uid].evidence_artifact_hashes,
        )
        for uid in sorted(quality_rejected)
    )
    coverage_request = CoverageRequest(
        chunks=request.chunks,
        proposed_survivors=proposed,
        strata=request.coverage_strata,
        redundancy_families=request.redundancy_families,
        similarities=request.similarities,
        exclusions=exclusions,
        provider_id=request.semantic_provider_id,
        provider_identity_sha256=request.semantic_provider_identity_sha256,
    )
    coverage = evaluate_coverage(coverage_request, semantic_provider)
    if coverage.status is CoverageStatus.ABSTAIN:
        blocked_evidence = tuple(sorted({gates.identity_sha256(), *gates.evidence_artifact_hashes}))
        return _retain_base(
            request,
            profile,
            _BaseOutcome(JointSelectionStatus.BLOCKED_RETAIN_BASE, "joint_coverage_abstained", blocked_evidence),
        )
    final = proposed | frozenset(coverage.protected_uids)
    representatives = {choice.family_id: choice.representative_uid for choice in coverage.family_representatives}
    family_by_uid = {uid: family for family in request.redundancy_families for uid in family.member_uids}
    traces: list[JointRemovalTrace] = []
    for uid in sorted(universe - final):
        quality = quality_by_uid[uid]
        if quality.decision is QualityEffectDecisionName.REJECT_CANDIDATE:
            traces.append(
                JointRemovalTrace(
                    uid,
                    "quality",
                    "stage_c_calibrated_quality_effect_candidate",
                    quality.reason_code,
                    None,
                    quality.evidence_artifact_hashes,
                )
            )
            continue
        family = family_by_uid.get(uid)
        if family is None or family.family_id not in representatives:
            raise JointSelectorContractError("Every non-Quality removal requires a final redundancy representative")
        trace_hashes = tuple(sorted({family.evidence_artifact_sha256, *coverage.evidence_artifact_hashes}))
        traces.append(
            JointRemovalTrace(
                uid,
                "redundancy",
                "candidate_redundancy_v2",
                "redundancy_family_nonrepresentative",
                representatives[family.family_id],
                trace_hashes,
            )
        )
    evidence_hashes = tuple(
        sorted(
            {
                *gates.evidence_artifact_hashes,
                gates.identity_sha256(),
                *coverage.evidence_artifact_hashes,
                *(artifact for decision in request.quality_decisions for artifact in decision.evidence_artifact_hashes),
            }
        )
    )
    parts = JointResultParts(
        JointSelectionStatus.CANDIDATE_MATERIALIZED,
        "joint_candidate_materialized",
        tuple(sorted(final)),
        tuple(traces),
        coverage,
        evidence_hashes,
    )
    return finalize_joint_result(request, profile, parts)


def evaluate_joint_selection(
    request: JointSelectionRequest,
    profile: JointProfileSpec,
    gates: JointGateBundle,
    semantic_provider: ProviderManifest,
) -> JointSelectionResult:
    match profile.name:
        case JointProfileName.BASE:
            return _retain_base(
                request,
                profile,
                _BaseOutcome(JointSelectionStatus.BASE_MATERIALIZED, "joint_base_validity_universe", ()),
            )
        case JointProfileName.HARD:
            gate_evidence = tuple(sorted({gates.identity_sha256(), *gates.evidence_artifact_hashes}))
            if not gates.hard_extension_ready or not profile.hard_extension_policy_ids:
                return _retain_base(
                    request,
                    profile,
                    _BaseOutcome(JointSelectionStatus.BLOCKED_RETAIN_BASE, "joint_hard_extension_not_ready", gate_evidence),
                )
            return _retain_base(
                request,
                profile,
                _BaseOutcome(JointSelectionStatus.BLOCKED_RETAIN_BASE, "joint_hard_extension_executor_missing", gate_evidence),
            )
        case JointProfileName.NORMAL:
            required_ready = gates.redundancy_ready and gates.quality_ready and gates.coverage_ready and gates.external_results_hidden
            if not required_ready:
                gate_evidence = tuple(sorted({gates.identity_sha256(), *gates.evidence_artifact_hashes}))
                return _retain_base(
                    request,
                    profile,
                    _BaseOutcome(JointSelectionStatus.BLOCKED_RETAIN_BASE, "joint_required_evidence_gate_blocked", gate_evidence),
                )
            return _normal_candidate(request, profile, gates, semantic_provider)
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "JointGateBundle",
    "JointGateOrigin",
    "JointProfileName",
    "JointSelectionRequest",
    "JointSelectionStatus",
    "evaluate_joint_selection",
    "load_current_joint_gates",
    "load_joint_profiles",
]
