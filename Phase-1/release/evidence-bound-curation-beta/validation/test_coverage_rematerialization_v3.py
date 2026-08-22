#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_contract import (
    CoverageChunk,
    CoverageRequest,
    CoverageStatus,
    CoverageStratum,
    CoverageView,
    ExclusionEvidence,
    ExclusionKind,
    FrozenSimilarity,
    RepresentativeFamily,
    StratumState,
)
from coverage_rematerialization import rematerialize_with_coverage
from coverage_redundancy_bridge import coverage_families_from_redundancy_plan
from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
)
from redundancy_mode_policy import RedundancyMode, build_redundancy_plan
from redundancy_v2 import RedundancySettings, RedundancyUnit


def _provider() -> ProviderManifest:
    return ProviderManifest(
        provider_id="semantic-fixture",
        role=ProviderRole.SEMANTIC,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.ACTIVE,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="multilingual-consensus-fixture",
        output_semantics="stable-semantic-strata-and-required-retain",
        supported_routes=("all",),
        supported_languages=("multilingual",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=CalibrationEvidence(
            artifact_path="calibration.json",
            artifact_sha256="1" * 64,
            scope_id="development",
        ),
        validation=ValidationEvidence(
            artifact_path="validation.json",
            artifact_sha256="2" * 64,
            scope_id="confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _request(provider: ProviderManifest, exclusions: tuple[ExclusionEvidence, ...] = ()) -> CoverageRequest:
    return CoverageRequest(
        chunks=(CoverageChunk("family-a", 10), CoverageChunk("family-b", 30), CoverageChunk("tail-ko", 20)),
        proposed_survivors=frozenset(),
        strata=(
            CoverageStratum("tail-hangul", CoverageView.SEMANTIC_SKILL, frozenset({"tail-ko"}), StratumState.STABLE, "3" * 64),
        ),
        redundancy_families=(
            RepresentativeFamily("family", frozenset({"family-a", "family-b"}), "4" * 64, "family-b"),
        ),
        similarities=(FrozenSimilarity("family-a", "family-b", 1.0, "5" * 64),),
        exclusions=exclusions,
        provider_id=provider.provider_id,
        provider_identity_sha256=provider.identity_sha256(),
    )


def test_veto_is_explicitly_rematerialized_and_rechecked() -> None:
    provider = _provider()
    result = rematerialize_with_coverage(_request(provider), provider)

    assert result.initial_decision.status is CoverageStatus.VETO_CANDIDATE
    assert result.required_retain_uids == ("family-b", "tail-ko")
    assert result.final_survivor_uids == ("family-b", "tail-ko")
    assert result.final_decision.status is CoverageStatus.PASS
    assert result.rematerialization_applied is True
    assert result.silent_restore is False
    assert result.final_decision.family_representatives[0].representative_uid == "family-b"
    assert tuple(
        (trace.chunk_uid, trace.group_id, trace.view)
        for trace in result.initial_decision.restoration_traces
    ) == (
        ("family-b", "family", CoverageView.REDUNDANCY_FAMILY),
        ("tail-ko", "tail-hangul", CoverageView.SEMANTIC_SKILL),
    )
    assert result.final_decision.restoration_traces == ()


def test_independently_excluded_content_is_never_restored() -> None:
    provider = _provider()
    exclusion = ExclusionEvidence(
        "tail-ko",
        ExclusionKind.VALIDITY_INVALID,
        "validity-v2",
        "validity_unrecoverable_encoding",
        ("6" * 64,),
    )

    result = rematerialize_with_coverage(_request(provider, (exclusion,)), provider)

    assert "tail-ko" not in result.final_survivor_uids
    assert "tail-ko" not in result.required_retain_uids
    assert "tail-hangul" in result.final_decision.permitted_extinctions


def test_redundancy_directional_representative_reaches_coverage_unchanged() -> None:
    short = (
        "A complete payload remains observable after deterministic family compaction "
        "and preserves every declared condition outcome and relation."
    )
    long = f"Introduction. {short} Appendix with additional verified context."
    plan = build_redundancy_plan(
        (RedundancyUnit("short", short), RedundancyUnit("long", long)),
        RedundancySettings(),
        RedundancyMode.HARD,
        exhaustive=True,
    )

    families = coverage_families_from_redundancy_plan(plan)

    assert len(families) == 1
    assert families[0].preferred_representative_uid == "long"
    assert families[0].member_uids == frozenset({"short", "long"})


if __name__ == "__main__":
    test_veto_is_explicitly_rematerialized_and_rechecked()
    test_independently_excluded_content_is_never_restored()
    test_redundancy_directional_representative_reaches_coverage_unchanged()
    print("[coverage-rematerialization-v3] explicit veto and recheck: pass")
