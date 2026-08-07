#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import assert_never


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_engine import (
    CoverageChunk,
    CoverageStratum,
    CoverageView,
    FrozenSimilarity,
    RepresentativeFamily,
    StratumState,
)
from joint_selector import (
    JointGateBundle,
    JointGateOrigin,
    JointProfileName,
    JointSelectionRequest,
    JointSelectionStatus,
    JointSelectorContractError,
    evaluate_joint_selection,
    load_current_joint_gates,
    load_joint_profiles,
)
from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
)
from quality_effect_calibration import EffectDirection
from quality_effect_engine import QualityEffectDecision, QualityEffectDecisionName


PROFILES = ROOT / "configs" / "joint_selector_profiles_v1.json"
POLICY_PROFILES = ROOT / "configs" / "policy_profiles.json"
CORE_REGISTRY = ROOT / "configs" / "core_policy_registry.json"


def _semantic_provider() -> ProviderManifest:
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
            artifact_path="fixture-calibration.json",
            artifact_sha256="a" * 64,
            scope_id="fixture-development",
        ),
        validation=ValidationEvidence(
            artifact_path="fixture-validation.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _quality(uid: str, decision: QualityEffectDecisionName) -> QualityEffectDecision:
    hashes = tuple(str(index) * 64 for index in range(1, 6))
    match decision:
        case QualityEffectDecisionName.REJECT_CANDIDATE:
            return QualityEffectDecision(
                decision,
                "quality_nonpositive_effect_supported",
                uid,
                EffectDirection.SUPPORTED_NONPOSITIVE,
                hashes,
            )
        case QualityEffectDecisionName.ELIGIBLE_KEEP:
            return QualityEffectDecision(
                decision,
                "quality_positive_effect_supported",
                uid,
                EffectDirection.SUPPORTED_POSITIVE,
                hashes,
            )
        case QualityEffectDecisionName.ABSTAIN_RETAIN:
            return QualityEffectDecision(decision, "quality_effect_uncertain", uid, EffectDirection.UNCERTAIN, hashes)
        case unreachable:
            assert_never(unreachable)


def _request(provider: ProviderManifest) -> JointSelectionRequest:
    chunks = tuple(CoverageChunk(uid, tokens) for uid, tokens in (("a", 10), ("b", 20), ("c", 30), ("d", 40), ("e", 50)))
    return JointSelectionRequest(
        chunks=chunks,
        redundancy_families=(RepresentativeFamily("family-ab", frozenset({"a", "b"}), "9" * 64),),
        quality_decisions=tuple(
            _quality(uid, decision)
            for uid, decision in (
                ("a", QualityEffectDecisionName.ELIGIBLE_KEEP),
                ("b", QualityEffectDecisionName.ABSTAIN_RETAIN),
                ("c", QualityEffectDecisionName.ELIGIBLE_KEEP),
                ("d", QualityEffectDecisionName.ABSTAIN_RETAIN),
                ("e", QualityEffectDecisionName.REJECT_CANDIDATE),
            )
        ),
        coverage_strata=(
            CoverageStratum("route-main", CoverageView.CONTENT_ROUTE, frozenset({"a", "b", "c"}), StratumState.STABLE, "c" * 64),
            CoverageStratum("skill-tail", CoverageView.SEMANTIC_SKILL, frozenset({"d", "e"}), StratumState.STABLE, "d" * 64),
        ),
        similarities=(
            FrozenSimilarity("a", "b", 1.0, "f" * 64),
            FrozenSimilarity("a", "c", 0.1, "f" * 64),
            FrozenSimilarity("b", "c", 0.1, "f" * 64),
            FrozenSimilarity("c", "d", 0.1, "f" * 64),
            FrozenSimilarity("d", "e", 0.2, "f" * 64),
        ),
        quality_provider_identity_sha256="1" * 64,
        semantic_provider_id=provider.provider_id,
        semantic_provider_identity_sha256=provider.identity_sha256(),
    )


def _gates(*, hard_ready: bool = False, quality_ready: bool = True) -> JointGateBundle:
    return JointGateBundle(
        origin=JointGateOrigin.CONTRACT_FIXTURE,
        redundancy_ready=True,
        quality_ready=quality_ready,
        coverage_ready=True,
        hard_extension_ready=hard_ready,
        external_results_hidden=True,
        evidence_artifact_hashes=("6" * 64, "7" * 64, "8" * 64),
    )


def test_base_is_the_unchanged_validity_passing_universe() -> None:
    provider = _semantic_provider()
    profiles = load_joint_profiles(PROFILES)
    result = evaluate_joint_selection(_request(provider), profiles.by_name(JointProfileName.BASE), _gates(), provider)

    assert result.status is JointSelectionStatus.BASE_MATERIALIZED
    assert result.selected_uids == ("a", "b", "c", "d", "e")
    assert result.removal_traces == ()
    assert result.may_mutate_active_runtime is False


def test_normal_applies_all_cores_atomically_with_complete_traces() -> None:
    provider = _semantic_provider()
    profile = load_joint_profiles(PROFILES).by_name(JointProfileName.NORMAL)
    first = evaluate_joint_selection(_request(provider), profile, _gates(), provider)
    second = evaluate_joint_selection(_request(provider), profile, _gates(), provider)

    assert first == second
    assert first.status is JointSelectionStatus.CANDIDATE_MATERIALIZED
    assert first.selected_uids == ("a", "c", "d")
    traces = {trace.chunk_uid: trace for trace in first.removal_traces}
    assert traces["b"].authority_core == "redundancy"
    assert traces["b"].representative_chunk_uid == "a"
    assert traces["e"].authority_core == "quality"
    assert traces["e"].reason_code == "quality_nonpositive_effect_supported"
    assert all(trace.evidence_artifact_hashes for trace in first.removal_traces)
    assert first.coverage_decision is not None
    assert first.coverage_required_retain_uids == ()
    assert first.coverage_rematerialization_applied is False
    assert first.profile_sha256 == profile.identity_sha256()
    assert first.input_sha256
    assert first.manifest_sha256
    assert provider.identity_sha256() in first.evidence_artifact_hashes


def test_missing_gate_discards_the_complete_candidate_and_retains_base() -> None:
    provider = _semantic_provider()
    profile = load_joint_profiles(PROFILES).by_name(JointProfileName.NORMAL)
    result = evaluate_joint_selection(_request(provider), profile, _gates(quality_ready=False), provider)

    assert result.status is JointSelectionStatus.BLOCKED_RETAIN_BASE
    assert result.reason_code == "joint_required_evidence_gate_blocked"
    assert result.selected_uids == ("a", "b", "c", "d", "e")
    assert result.removal_traces == ()


def test_hard_profile_fails_closed_until_block_8_selects_an_extension() -> None:
    provider = _semantic_provider()
    profile = load_joint_profiles(PROFILES).by_name(JointProfileName.HARD)
    result = evaluate_joint_selection(_request(provider), profile, _gates(hard_ready=False), provider)

    assert result.status is JointSelectionStatus.BLOCKED_RETAIN_BASE
    assert result.reason_code == "joint_hard_extension_not_ready"
    assert result.selected_uids == ("a", "b", "c", "d", "e")


def test_target_aware_profiles_do_not_replace_the_active_structural_profile() -> None:
    target_aware = load_joint_profiles(PROFILES)
    current = json.loads(POLICY_PROFILES.read_text(encoding="utf-8"))
    core_registry = json.loads(CORE_REGISTRY.read_text(encoding="utf-8"))
    current_by_id = {profile["id"]: profile for profile in current["profiles"]}
    joint_contract = core_registry["core_decision_contracts"]["candidate_joint_selector_v1"]

    assert target_aware.runtime_activation is False
    assert target_aware.post_run_override_allowed is False
    assert target_aware.benchmark_feedback_allowed is False
    assert target_aware.source_selection_axis is False
    assert current_by_id["normal_structural_v1"]["status"] == "active"
    assert current_by_id["normal_structural_v1"]["selector"]["kind"] == "reason_coded_structural_and_teacher_evidence"
    assert current["target_aware_candidate_profile_registry"] == "configs/joint_selector_profiles_v1.json"
    assert joint_contract["orchestration_role"] == "atomic_evidence_only_combination_not_a_fifth_core"
    assert joint_contract["runtime_activation"] is False


def test_current_gates_are_derived_from_frozen_configs_and_remain_blocked() -> None:
    gates = load_current_joint_gates(ROOT)

    assert gates.origin is JointGateOrigin.FROZEN_REGISTRY
    assert gates.redundancy_ready is False
    assert gates.quality_ready is False
    assert gates.coverage_ready is False
    assert gates.hard_extension_ready is False
    assert len(gates.evidence_artifact_hashes) == 3


def test_profile_loader_rejects_structural_policy_drift() -> None:
    payload = json.loads(PROFILES.read_text(encoding="utf-8"))
    normal = next(profile for profile in payload["profiles"] if profile["name"] == "normal")
    normal["required_policy_ids"] = []
    drift_rejected = False
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "profiles.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_joint_profiles(path)
        except JointSelectorContractError:
            drift_rejected = True
    assert drift_rejected is True


if __name__ == "__main__":
    test_base_is_the_unchanged_validity_passing_universe()
    test_normal_applies_all_cores_atomically_with_complete_traces()
    test_missing_gate_discards_the_complete_candidate_and_retains_base()
    test_hard_profile_fails_closed_until_block_8_selects_an_extension()
    test_target_aware_profiles_do_not_replace_the_active_structural_profile()
    test_current_gates_are_derived_from_frozen_configs_and_remain_blocked()
    test_profile_loader_rejects_structural_policy_drift()
    print("[joint-selector-v1] atomic target-aware profile contract: pass")
