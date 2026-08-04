#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_engine import (
    CoverageChunk,
    CoverageContractError,
    CoverageRequest,
    CoverageStatus,
    CoverageStratum,
    CoverageView,
    ExclusionEvidence,
    ExclusionKind,
    FrozenSimilarity,
    RepresentativeFamily,
    StratumState,
    evaluate_coverage,
)
from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
    load_provider_registry,
)


CONTRACT = ROOT / "configs" / "coverage_engine_v2.json"
PROVIDER_REGISTRY = ROOT / "configs" / "model_provider_registry_v1.json"
POLICY_REGISTRY = ROOT / "configs" / "core_policy_registry.json"
POLICY_CARDS = ROOT / "configs" / "policy_cards.json"


def _active_provider() -> ProviderManifest:
    return ProviderManifest(
        provider_id="fixture-semantic-provider",
        role=ProviderRole.SEMANTIC,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.ACTIVE,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="fixture-frozen-similarity-v1",
        output_semantics="frozen-pairwise-similarity-and-stable-strata",
        supported_routes=("fixture",),
        supported_languages=("fixture",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=CalibrationEvidence(
            artifact_path="validation/frozen_contracts/coverage-calibration.json",
            artifact_sha256="a" * 64,
            scope_id="fixture-development",
        ),
        validation=ValidationEvidence(
            artifact_path="validation/frozen_contracts/coverage-confirmatory.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _request(provider: ProviderManifest, exclusions: tuple[ExclusionEvidence, ...] = ()) -> CoverageRequest:
    chunks = tuple(CoverageChunk(uid, tokens) for uid, tokens in (("a", 10), ("b", 20), ("c", 30), ("d", 40), ("e", 50)))
    strata = (
        CoverageStratum("route-main", CoverageView.CONTENT_ROUTE, frozenset({"a", "b", "c"}), StratumState.STABLE, "c" * 64),
        CoverageStratum("skill-tail", CoverageView.SEMANTIC_SKILL, frozenset({"d", "e"}), StratumState.STABLE, "d" * 64),
        CoverageStratum("uncertain-mixed", CoverageView.UNCERTAIN_INTERSECTION, frozenset({"a", "b"}), StratumState.UNCERTAIN, "e" * 64),
    )
    similarities = (
        FrozenSimilarity("a", "b", 1.0, "f" * 64),
        FrozenSimilarity("a", "c", 0.1, "f" * 64),
        FrozenSimilarity("b", "c", 0.1, "f" * 64),
        FrozenSimilarity("c", "d", 0.1, "f" * 64),
        FrozenSimilarity("c", "e", 0.5, "f" * 64),
        FrozenSimilarity("d", "e", 0.2, "f" * 64),
    )
    return CoverageRequest(
        chunks=chunks,
        proposed_survivors=frozenset({"c"}),
        strata=strata,
        redundancy_families=(RepresentativeFamily("family-ab", frozenset({"a", "b"}), "9" * 64),),
        similarities=similarities,
        exclusions=exclusions,
        provider_id=provider.provider_id,
        provider_identity_sha256=provider.identity_sha256(),
    )


def test_extinction_and_family_constraints_produce_nonmutating_veto_candidates() -> None:
    provider = _active_provider()
    decision = evaluate_coverage(_request(provider), provider)

    assert decision.status is CoverageStatus.VETO_CANDIDATE
    assert decision.protected_uids == ("a", "d")
    assert decision.family_representatives[0].representative_uid == "a"
    assert decision.family_representatives[0].selection_method == "facility_location_marginal_gain_then_uid"
    assert decision.extinct_before_protection == ("skill-tail", "uncertain-mixed")
    assert decision.permitted_extinctions == ()
    assert decision.may_mutate_curated_membership is False
    assert decision.fixed_quota_used is False
    assert decision.source_identity_used is False
    assert decision.benchmark_outcomes_read is False
    assert decision.utility_read is False


def test_extinction_is_permitted_only_when_every_member_has_independent_evidence() -> None:
    provider = _active_provider()
    exclusions = tuple(
        ExclusionEvidence(uid, ExclusionKind.VALIDITY_INVALID, "validity-v2", f"validity_{uid}", ("8" * 64,))
        for uid in ("d", "e")
    )
    decision = evaluate_coverage(_request(provider, exclusions), provider)

    assert "skill-tail" in decision.permitted_extinctions
    assert "d" not in decision.protected_uids
    assert "e" not in decision.protected_uids


def test_partial_exclusion_protects_the_remaining_eligible_member() -> None:
    provider = _active_provider()
    exclusion = ExclusionEvidence(
        "d",
        ExclusionKind.QUALITY_SUPPORTED_NONPOSITIVE,
        "stage_c_calibrated_quality_effect_candidate",
        "quality_nonpositive_effect_supported",
        tuple(str(index) * 64 for index in range(1, 6)),
    )
    decision = evaluate_coverage(_request(provider, (exclusion,)), provider)

    assert "e" in decision.protected_uids
    assert "skill-tail" not in decision.permitted_extinctions


def test_current_audit_only_semantic_provider_abstains_without_veto_candidates() -> None:
    registry = load_provider_registry(PROVIDER_REGISTRY)
    provider = next(item for item in registry.providers if item.role is ProviderRole.SEMANTIC)
    decision = evaluate_coverage(_request(provider), provider)

    assert provider.lifecycle is ProviderLifecycle.AUDIT_ONLY
    assert decision.status is CoverageStatus.ABSTAIN
    assert decision.reason_code == "coverage_semantic_provider_not_active"
    assert decision.protected_uids == ()


def test_provider_identity_change_and_malformed_universe_fail_closed() -> None:
    provider = _active_provider()
    mismatch = evaluate_coverage(replace(_request(provider), provider_identity_sha256="0" * 64), provider)
    assert mismatch.status is CoverageStatus.ABSTAIN
    assert mismatch.reason_code == "coverage_semantic_provider_identity_mismatch"

    malformed_raised = False
    try:
        replace(_request(provider), proposed_survivors=frozenset({"missing"}))
    except CoverageContractError:
        malformed_raised = True
    assert malformed_raised is True

    forged_quality_raised = False
    try:
        ExclusionEvidence(
            "d",
            ExclusionKind.QUALITY_SUPPORTED_NONPOSITIVE,
            "stage_c_calibrated_quality_effect_candidate",
            "quality_nonpositive_effect_supported",
            ("7" * 64,),
        )
    except CoverageContractError:
        forged_quality_raised = True
    assert forged_quality_raised is True


def test_report_is_a_vector_without_intrinsic_coverage_score() -> None:
    provider = _active_provider()
    decision = evaluate_coverage(_request(provider), provider)
    by_view = {report.view: report for report in decision.view_reports}

    assert CoverageView.SEMANTIC_SKILL in by_view
    assert by_view[CoverageView.SEMANTIC_SKILL].proposed_support_recall == 0.0
    assert by_view[CoverageView.SEMANTIC_SKILL].protected_support_recall == 1.0
    assert decision.token_report.raw_tokens == 150
    assert decision.token_report.proposed_tokens == 30
    assert decision.token_report.protected_tokens == 80
    assert 0.0 <= decision.nearest_representative_radius <= 1.0
    assert decision.effective_sample_size > 0.0
    assert not hasattr(decision, "overall_coverage_score")


def test_contract_blocks_quota_source_and_real_activation() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "semantic_v3_implemented_empirical_gates_blocked"
    assert contract["runtime_activation"] is False
    assert contract["fixed_domain_quota"] is False
    assert contract["source_selection_axis"] is False
    assert contract["single_coverage_score"] is False
    assert contract["current_semantic_provider_state"] == "audit_only"
    assert contract["normal_hard_coverage_invariants_identical"] is True
    assert contract["explicit_required_retain_rematerialization"] is True


def test_candidate_registry_preserves_the_active_materialization_guard() -> None:
    registry = json.loads(POLICY_REGISTRY.read_text(encoding="utf-8"))
    cards = json.loads(POLICY_CARDS.read_text(encoding="utf-8"))
    policies = {policy["id"]: policy for policy in registry["policies"]}
    cards_by_id = {card["id"]: card for card in cards["cards"]}
    candidate = policies["stage_c_coverage_support_candidate"]
    active_guard = policies["stage_c_coverage_guard"]
    profile = registry["runtime_profile_authorization"]["normal_structural_v1"]

    assert candidate["status"] == "candidate"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_mutate_membership"
    assert candidate["version"] == cards_by_id[candidate["policy_card_id"]]["version"]
    assert candidate["runtime_implementation"] == cards_by_id[candidate["policy_card_id"]]["runtime_implementation"]
    assert candidate["id"] in profile["excluded_policy_ids"]
    assert active_guard["status"] == "active"
    assert active_guard["id"] in profile["enabled_policy_ids"]


if __name__ == "__main__":
    test_extinction_and_family_constraints_produce_nonmutating_veto_candidates()
    test_extinction_is_permitted_only_when_every_member_has_independent_evidence()
    test_partial_exclusion_protects_the_remaining_eligible_member()
    test_current_audit_only_semantic_provider_abstains_without_veto_candidates()
    test_provider_identity_change_and_malformed_universe_fail_closed()
    test_report_is_a_vector_without_intrinsic_coverage_score()
    test_contract_blocks_quota_source_and_real_activation()
    test_candidate_registry_preserves_the_active_materialization_guard()
    print("[coverage-engine-v2] support veto and representative contract: pass")
