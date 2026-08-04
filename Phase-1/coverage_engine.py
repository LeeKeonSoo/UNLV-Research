from __future__ import annotations

from collections import defaultdict
from typing import assert_never

from coverage_contract import (
    CoverageChunk,
    CoverageContractError,
    CoverageDecision,
    CoverageRequest,
    CoverageStatus,
    CoverageStratum,
    CoverageView,
    CoverageViewReport,
    ExclusionEvidence,
    ExclusionKind,
    FrozenSimilarity,
    RepresentativeChoice,
    RepresentativeFamily,
    StratumState,
    TokenMassReport,
)
from coverage_metrics import (
    choose_by_marginal_gain,
    effective_sample_size,
    jensen_shannon_divergence,
    nearest_representative_radius,
)
from model_provider_contract import ProviderLifecycle, ProviderManifest, ProviderRole


def _token_report(request: CoverageRequest, selected: frozenset[str]) -> TokenMassReport:
    raw = sum(chunk.token_count for chunk in request.chunks)
    proposed = sum(chunk.token_count for chunk in request.chunks if chunk.uid in request.proposed_survivors)
    protected = sum(chunk.token_count for chunk in request.chunks if chunk.uid in selected)
    return TokenMassReport(raw, proposed, protected, proposed / raw, protected / raw)


def _abstain(request: CoverageRequest, reason_code: str) -> CoverageDecision:
    return CoverageDecision(
        status=CoverageStatus.ABSTAIN,
        reason_code=reason_code,
        protected_uids=(),
        family_representatives=(),
        extinct_before_protection=(),
        permitted_extinctions=(),
        view_reports=(),
        token_report=_token_report(request, request.proposed_survivors),
        nearest_representative_radius=1.0,
        effective_sample_size=effective_sample_size(request.chunks, request.proposed_survivors),
        evidence_artifact_hashes=(),
    )


def _provider_gate(request: CoverageRequest, provider: ProviderManifest) -> str | None:
    match provider.role:
        case ProviderRole.SEMANTIC:
            pass
        case ProviderRole.QUALITY | ProviderRole.DIAGNOSTIC_VALIDITY | ProviderRole.CONTENT_ROUTER:
            return "coverage_provider_role_mismatch"
        case unreachable:
            assert_never(unreachable)
    match provider.lifecycle:
        case ProviderLifecycle.ACTIVE:
            pass
        case (
            ProviderLifecycle.AUDIT_ONLY
            | ProviderLifecycle.CALIBRATED
            | ProviderLifecycle.DEVELOPMENT_VALIDATED
            | ProviderLifecycle.CONFIRMATORY_VALIDATED
            | ProviderLifecycle.RETIRED
        ):
            return "coverage_semantic_provider_not_active"
        case unreachable:
            assert_never(unreachable)
    if not provider.policy_contribution_authority:
        return "coverage_semantic_provider_not_active"
    if request.provider_id != provider.provider_id or request.provider_identity_sha256 != provider.identity_sha256():
        return "coverage_semantic_provider_identity_mismatch"
    return None


def _view_reports(
    request: CoverageRequest,
    eligible: frozenset[str],
    protected: frozenset[str],
) -> tuple[CoverageViewReport, ...]:
    token_by_uid = {chunk.uid: chunk.token_count for chunk in request.chunks}
    by_view: dict[CoverageView, list[CoverageStratum]] = defaultdict(list)
    for stratum in request.strata:
        if stratum.member_uids & eligible:
            by_view[stratum.view].append(stratum)
    reports: list[CoverageViewReport] = []
    for view in sorted(by_view, key=lambda item: item.value):
        strata = sorted(by_view[view], key=lambda item: item.stratum_id)
        proposed_covered = sum(bool(stratum.member_uids & request.proposed_survivors) for stratum in strata)
        protected_covered = sum(bool(stratum.member_uids & protected) for stratum in strata)
        raw_mass = tuple(float(sum(token_by_uid[uid] for uid in stratum.member_uids & eligible)) for stratum in strata)
        protected_mass = tuple(float(sum(token_by_uid[uid] for uid in stratum.member_uids & protected)) for stratum in strata)
        count = len(strata)
        reports.append(
            CoverageViewReport(
                view=view,
                target_strata=count,
                proposed_covered_strata=proposed_covered,
                protected_covered_strata=protected_covered,
                proposed_support_recall=proposed_covered / count,
                protected_support_recall=protected_covered / count,
                token_mass_jensen_shannon_divergence=jensen_shannon_divergence(raw_mass, protected_mass),
            )
        )
    return tuple(reports)


def _evidence_hashes(request: CoverageRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                request.provider_identity_sha256,
                *(stratum.evidence_artifact_sha256 for stratum in request.strata),
                *(family.evidence_artifact_sha256 for family in request.redundancy_families),
                *(edge.evidence_artifact_sha256 for edge in request.similarities),
                *(artifact for exclusion in request.exclusions for artifact in exclusion.evidence_artifact_hashes),
            }
        )
    )


def evaluate_coverage(request: CoverageRequest, provider: ProviderManifest) -> CoverageDecision:
    provider_failure = _provider_gate(request, provider)
    if provider_failure is not None:
        return _abstain(request, provider_failure)

    excluded = frozenset(exclusion.chunk_uid for exclusion in request.exclusions)
    eligible = frozenset(chunk.uid for chunk in request.chunks) - excluded
    eligible_order = tuple(sorted(eligible))
    working = set(request.proposed_survivors)
    protected: set[str] = set()
    permitted: set[str] = set()
    extinct_before = tuple(
        sorted(
            stratum.stratum_id
            for stratum in request.strata
            if stratum.member_uids & eligible and not stratum.member_uids & request.proposed_survivors
        )
    )
    representatives: list[RepresentativeChoice] = []

    for family in sorted(request.redundancy_families, key=lambda item: item.family_id):
        candidates = family.member_uids & eligible
        if not candidates:
            permitted.add(f"family:{family.family_id}")
            continue
        surviving_candidates = candidates & working
        choice_pool = frozenset(surviving_candidates or candidates)
        preferred = family.preferred_representative_uid
        if preferred is not None and preferred in choice_pool:
            representative = preferred
            gain = 0.0
            selection_method = "stage_b_directional_preference"
        else:
            representative, gain = choose_by_marginal_gain(
                choice_pool, frozenset(working), eligible_order, request.similarities
            )
            selection_method = "facility_location_marginal_gain_then_uid"
        representatives.append(
            RepresentativeChoice(family.family_id, representative, gain, selection_method)
        )
        if not surviving_candidates:
            working.add(representative)
            protected.add(representative)

    for stratum in sorted(request.strata, key=lambda item: item.stratum_id):
        candidates = stratum.member_uids & eligible
        if not candidates:
            permitted.add(stratum.stratum_id)
            continue
        if candidates & working:
            continue
        representative, _ = choose_by_marginal_gain(frozenset(candidates), frozenset(working), eligible_order, request.similarities)
        working.add(representative)
        protected.add(representative)

    final_survivors = frozenset(working)
    status = CoverageStatus.VETO_CANDIDATE if protected else CoverageStatus.PASS
    reason = "coverage_support_veto_candidates_required" if protected else "coverage_constraints_satisfied"
    return CoverageDecision(
        status=status,
        reason_code=reason,
        protected_uids=tuple(sorted(protected)),
        family_representatives=tuple(representatives),
        extinct_before_protection=extinct_before,
        permitted_extinctions=tuple(sorted(permitted)),
        view_reports=_view_reports(request, eligible, final_survivors),
        token_report=_token_report(request, final_survivors),
        nearest_representative_radius=nearest_representative_radius(eligible_order, final_survivors, request.similarities),
        effective_sample_size=effective_sample_size(request.chunks, final_survivors),
        evidence_artifact_hashes=_evidence_hashes(request),
    )


__all__ = [
    "CoverageChunk",
    "CoverageContractError",
    "CoverageRequest",
    "CoverageStatus",
    "CoverageStratum",
    "CoverageView",
    "ExclusionEvidence",
    "ExclusionKind",
    "FrozenSimilarity",
    "RepresentativeFamily",
    "StratumState",
    "evaluate_coverage",
]
