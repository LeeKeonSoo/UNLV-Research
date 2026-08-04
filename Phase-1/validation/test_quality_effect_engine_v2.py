#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_provider_contract import (
    CalibrationEvidence,
    ProviderLifecycle,
    ProviderManifest,
    ProviderRole,
    ValidationEvidence,
    load_provider_registry,
)
from quality_effect_engine import (
    EffectCalibrationBundle,
    EffectInterval,
    EvidenceBin,
    QualityEffectContractError,
    QualityEffectObservation,
    QualityEvaluationRequest,
    calibrate_effect_bins,
    evaluate_quality_effect,
)


FIXTURES = ROOT / "validation" / "fixtures" / "quality_effect_engine_cases.json"
CONTRACT = ROOT / "configs" / "quality_effect_engine_v2.json"
PROVIDER_REGISTRY = ROOT / "configs" / "model_provider_registry_v1.json"


def _active_provider(supported_routes: tuple[str, ...] = ("general_prose", "code_artifact")) -> ProviderManifest:
    return ProviderManifest(
        provider_id="fixture-quality-provider",
        role=ProviderRole.QUALITY,
        provider_type="deterministic",
        lifecycle=ProviderLifecycle.ACTIVE,
        artifacts=(),
        tokenizer_id=None,
        tokenizer_revision=None,
        normalization="fixture-bin-assignment-v1",
        output_semantics="frozen-evidence-bin-id",
        supported_routes=supported_routes,
        supported_languages=("fixture",),
        policy_contribution_authority=True,
        direct_deletion_authority=False,
        calibration=CalibrationEvidence(
            artifact_path="validation/frozen_contracts/quality-effect.json",
            artifact_sha256="a" * 64,
            scope_id="fixture-development",
        ),
        validation=ValidationEvidence(
            artifact_path="validation/frozen_contracts/quality-effect-confirmatory.json",
            artifact_sha256="b" * 64,
            scope_id="fixture-confirmatory",
            three_seed_natural_budget_complete=True,
        ),
    )


def _bundle(provider: ProviderManifest | None = None) -> EffectCalibrationBundle:
    selected_provider = provider or _active_provider()
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    bins = tuple(
        EvidenceBin(
            route=row["route"],
            bin_id=row["bin_id"],
            bin_order=row["bin_order"],
            development=EffectInterval(**row["development"]),
            heldout=EffectInterval(**row["heldout"]),
            artifact_sha256="c" * 64,
        )
        for row in payload["bins"]
    )
    return EffectCalibrationBundle(
        provider_id=selected_provider.provider_id,
        provider_identity_sha256=selected_provider.identity_sha256(),
        effect_metric_id=payload["effect_metric_id"],
        effect_metric_artifact_sha256=payload["effect_metric_artifact_sha256"],
        common_baseline_artifact_sha256=payload["common_baseline_artifact_sha256"],
        bins=bins,
        provider_training_source_groups=frozenset(payload["provider_training_source_groups"]),
        development_source_groups=frozenset(payload["development_source_groups"]),
        heldout_source_groups=frozenset(payload["heldout_source_groups"]),
        all_arms_share_common_baseline=True,
        common_baseline_disjoint_from_all_bins=True,
        external_results_hidden=True,
        provider_bias_stress_passed=True,
        route_holdout_stress_passed=True,
    )


def _request(provider: ProviderManifest, bin_id: str, route_state: str = "routed") -> QualityEvaluationRequest:
    return QualityEvaluationRequest(
        observation=QualityEffectObservation(
            chunk_uid="chunk-1",
            route_state=route_state,
            route="general_prose" if route_state == "routed" else None,
            provider_id=provider.provider_id if route_state == "routed" else None,
            provider_identity_sha256=provider.identity_sha256() if route_state == "routed" else None,
            bin_id=bin_id if route_state == "routed" else None,
            observation_artifact_sha256="d" * 64 if route_state == "routed" else None,
        )
    )


def test_calibration_uses_measured_effects_without_weighted_quality_formula() -> None:
    report = calibrate_effect_bins(_bundle())

    assert report.passed is True
    assert report.weighted_quality_formula_used is False
    assert report.target_retention_fraction_used is False
    assert report.benchmark_outcomes_used is False
    assert report.effect_unit == "risk_reduction_per_target_token"
    assert report.common_baseline_artifact_sha256 == "f" * 64
    assert report.failed_gates == ()
    by_id = {effect.bin_id: effect for effect in report.bins}
    assert by_id["general-low"].direction.value == "supported_nonpositive"
    assert by_id["general-mid"].direction.value == "uncertain"
    assert by_id["general-high"].direction.value == "supported_positive"


def test_quality_decision_is_uncertainty_aware_and_never_mutates_runtime() -> None:
    provider = _active_provider()
    report = calibrate_effect_bins(_bundle(provider))

    low = evaluate_quality_effect(_request(provider, "general-low"), report, provider)
    middle = evaluate_quality_effect(_request(provider, "general-mid"), report, provider)
    high = evaluate_quality_effect(_request(provider, "general-high"), report, provider)

    assert low.decision.value == "reject_candidate"
    assert middle.decision.value == "abstain_retain"
    assert high.decision.value == "eligible_keep"
    assert all(not decision.may_mutate_curated_membership for decision in (low, middle, high))
    assert all(not decision.benchmark_outcomes_read for decision in (low, middle, high))
    assert all(not decision.utility_read for decision in (low, middle, high))
    assert set(low.evidence_artifact_hashes) == {
        provider.identity_sha256(),
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
    }


def test_unknown_mixed_ood_and_provider_identity_changes_retain() -> None:
    provider = _active_provider()
    report = calibrate_effect_bins(_bundle(provider))
    unknown = evaluate_quality_effect(_request(provider, "general-low", "unknown"), report, provider)
    changed = _request(provider, "general-low")
    changed_observation = QualityEffectObservation(
        chunk_uid=changed.observation.chunk_uid,
        route_state="routed",
        route="general_prose",
        provider_id=provider.provider_id,
        provider_identity_sha256="e" * 64,
        bin_id="general-low",
        observation_artifact_sha256="d" * 64,
    )
    mismatch = evaluate_quality_effect(QualityEvaluationRequest(changed_observation), report, provider)

    assert unknown.decision.value == "abstain_retain"
    assert unknown.reason_code == "quality_route_uncertain"
    assert mismatch.decision.value == "abstain_retain"
    assert mismatch.reason_code == "quality_provider_identity_mismatch"


def test_runtime_experiment_panel_does_not_activate_legacy_effect_selector() -> None:
    registry = load_provider_registry(PROVIDER_REGISTRY)
    provider = next(item for item in registry.providers if item.role is ProviderRole.QUALITY)
    report = calibrate_effect_bins(_bundle(provider))
    decision = evaluate_quality_effect(_request(provider, "general-low"), report, provider)

    assert provider.lifecycle is ProviderLifecycle.RUNTIME_EXPERIMENT
    assert decision.decision.value == "abstain_retain"
    assert decision.reason_code == "quality_provider_not_active"


def test_provider_cannot_borrow_an_unsupported_route() -> None:
    provider = _active_provider(("code_artifact",))
    report = calibrate_effect_bins(_bundle(provider))
    decision = evaluate_quality_effect(_request(provider, "general-low"), report, provider)

    assert decision.decision.value == "abstain_retain"
    assert decision.reason_code == "quality_provider_route_unsupported"


def test_calibration_fails_closed_on_leakage_or_nonmonotonic_holdout() -> None:
    clean = _bundle()
    leaked = EffectCalibrationBundle(
        provider_id=clean.provider_id,
        provider_identity_sha256=clean.provider_identity_sha256,
        effect_metric_id=clean.effect_metric_id,
        effect_metric_artifact_sha256=clean.effect_metric_artifact_sha256,
        common_baseline_artifact_sha256=clean.common_baseline_artifact_sha256,
        bins=clean.bins,
        provider_training_source_groups=clean.provider_training_source_groups,
        development_source_groups=clean.development_source_groups,
        heldout_source_groups=clean.heldout_source_groups,
        all_arms_share_common_baseline=True,
        common_baseline_disjoint_from_all_bins=True,
        external_results_hidden=False,
        provider_bias_stress_passed=True,
        route_holdout_stress_passed=True,
    )
    report = calibrate_effect_bins(leaked)
    unshared = replace(clean, all_arms_share_common_baseline=False)
    unshared_report = calibrate_effect_bins(unshared)
    high = next(effect for effect in clean.bins if effect.bin_id == "general-high")
    nonmonotonic_high = replace(
        high,
        heldout=EffectInterval(point=-0.01, lower=-0.06, upper=0.06, samples=9),
    )
    nonmonotonic = replace(
        clean,
        bins=tuple(nonmonotonic_high if effect.bin_id == "general-high" else effect for effect in clean.bins),
    )
    nonmonotonic_report = calibrate_effect_bins(nonmonotonic)

    assert report.passed is False
    assert "external_feedback_leakage" in report.failed_gates
    assert "common_baseline_not_shared" in unshared_report.failed_gates
    assert nonmonotonic_report.passed is False
    assert "heldout_nonmonotonic:general_prose" in nonmonotonic_report.failed_gates


def test_calibration_requires_three_bins_and_disjoint_nonempty_splits() -> None:
    clean = _bundle()
    too_few_bins = tuple(effect for effect in clean.bins if effect.bin_id != "general-high")
    invalid_specs = (
        (too_few_bins, clean.heldout_source_groups),
        (clean.bins, clean.development_source_groups),
    )

    for bins, heldout_groups in invalid_specs:
        contract_error_raised = False
        try:
            EffectCalibrationBundle(
                provider_id=clean.provider_id,
                provider_identity_sha256=clean.provider_identity_sha256,
                effect_metric_id=clean.effect_metric_id,
                effect_metric_artifact_sha256=clean.effect_metric_artifact_sha256,
                common_baseline_artifact_sha256=clean.common_baseline_artifact_sha256,
                bins=bins,
                provider_training_source_groups=clean.provider_training_source_groups,
                development_source_groups=clean.development_source_groups,
                heldout_source_groups=heldout_groups,
                all_arms_share_common_baseline=clean.all_arms_share_common_baseline,
                common_baseline_disjoint_from_all_bins=clean.common_baseline_disjoint_from_all_bins,
                external_results_hidden=clean.external_results_hidden,
                provider_bias_stress_passed=clean.provider_bias_stress_passed,
                route_holdout_stress_passed=clean.route_holdout_stress_passed,
            )
        except QualityEffectContractError:
            contract_error_raised = True
        assert contract_error_raised is True


def test_contract_reports_real_empirical_blockers() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "block_5_engine_fixture_validated_empirical_gates_blocked"
    assert contract["runtime_activation"] is False
    assert contract["current_quality_provider_state"] == "audit_only"
    assert contract["all_registered_routes_empirically_ready"] is False
    assert contract["unknown_mixed_ood_action"] == "abstain_retain"


if __name__ == "__main__":
    test_calibration_uses_measured_effects_without_weighted_quality_formula()
    test_quality_decision_is_uncertainty_aware_and_never_mutates_runtime()
    test_unknown_mixed_ood_and_provider_identity_changes_retain()
    test_runtime_experiment_panel_does_not_activate_legacy_effect_selector()
    test_provider_cannot_borrow_an_unsupported_route()
    test_calibration_fails_closed_on_leakage_or_nonmonotonic_holdout()
    test_calibration_requires_three_bins_and_disjoint_nonempty_splits()
    test_contract_reports_real_empirical_blockers()
    print("[quality-effect-engine-v2] calibrated bin effects and fail-closed lifecycle: pass")
