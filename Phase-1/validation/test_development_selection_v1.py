from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_selection import (
    ArmRole,
    CorpusDomain,
    CorpusScenario,
    CorpusSliceEvidence,
    DevelopmentArm,
    DevelopmentGateEvidence,
    DevelopmentSelectionBundle,
    DevelopmentSelectionContractError,
    DevelopmentSelectionStatus,
    load_development_protocol,
    select_development_profiles,
)
from development_selection_preflight import evaluate_current_development_preflight


HASH = hashlib.sha256
NORMAL_POLICIES = (
    "candidate_redundancy_v2",
    "stage_c_calibrated_quality_effect_candidate",
    "stage_c_coverage_support_candidate",
)


def _digest(value: str) -> str:
    return HASH(value.encode()).hexdigest()


def _slices() -> tuple[CorpusSliceEvidence, ...]:
    return tuple(
        CorpusSliceEvidence(
            slice_id=f"{domain.value}-{scenario.value}",
            domain=domain,
            scenario=scenario,
            record_count=400,
            record_ids_artifact_sha256=_digest(f"records-{domain.value}-{scenario.value}"),
            normalized_text_hashes_artifact_sha256=_digest(f"texts-{domain.value}-{scenario.value}"),
            confirmatory_record_overlap_count=0,
            confirmatory_text_overlap_count=0,
            confirmatory_source_snapshot_overlap_count=0,
            confirmatory_time_overlap_count=0,
            benchmark_exclusion_passed=True,
            evidence_artifact_sha256=_digest(f"slice-{domain.value}-{scenario.value}"),
        )
        for domain in CorpusDomain
        for scenario in CorpusScenario
    )


def _gates() -> DevelopmentGateEvidence:
    return DevelopmentGateEvidence(
        redundancy_ready=True,
        quality_ready=True,
        coverage_ready=True,
        external_results_hidden=True,
        clean_control_count=500,
        clean_control_false_positives=0,
        provider_bias_stress_passed=True,
        route_holdout_stress_passed=True,
        evidence_artifact_hashes=(_digest("r"), _digest("q"), _digest("c")),
    )


def _arms() -> tuple[DevelopmentArm, ...]:
    return (
        DevelopmentArm(
            arm_id="normal-candidate",
            role=ArmRole.NORMAL,
            required_policy_ids=NORMAL_POLICIES,
            hard_extension_policy_ids=(),
            exact_natural_tokens=820_000,
            development_gain_lcb_per_token=0.03,
            effect_metric_id="development-risk-reduction-v1",
            effect_metric_artifact_sha256=_digest("metric"),
            common_baseline_artifact_sha256=_digest("common-baseline"),
            all_sensitivity_arms_share_one_common_baseline=True,
            common_baseline_disjoint_from_all_arms=True,
            development_and_heldout_effect_arms_disjoint=True,
            maximum_coverage_js_divergence=0.02,
            minimum_support_recall=1.0,
            extinct_supported_strata=0,
            unknown_mixed_extinct_strata=0,
            representative_linkage_complete=True,
            valid_residuals_complete=True,
            removal_trace_count=120,
            complete_removal_trace_count=120,
            input_manifest_sha256=_digest("input"),
            evidence_artifact_hashes=(_digest("normal"),),
        ),
        DevelopmentArm(
            arm_id="hard-dominated",
            role=ArmRole.HARD_EXTENSION,
            required_policy_ids=NORMAL_POLICIES,
            hard_extension_policy_ids=("stage_c_repeated_span_template_candidate",),
            exact_natural_tokens=710_000,
            development_gain_lcb_per_token=0.01,
            effect_metric_id="development-risk-reduction-v1",
            effect_metric_artifact_sha256=_digest("metric"),
            common_baseline_artifact_sha256=_digest("common-baseline"),
            all_sensitivity_arms_share_one_common_baseline=True,
            common_baseline_disjoint_from_all_arms=True,
            development_and_heldout_effect_arms_disjoint=True,
            maximum_coverage_js_divergence=0.04,
            minimum_support_recall=1.0,
            extinct_supported_strata=0,
            unknown_mixed_extinct_strata=0,
            representative_linkage_complete=True,
            valid_residuals_complete=True,
            removal_trace_count=180,
            complete_removal_trace_count=180,
            input_manifest_sha256=_digest("input"),
            evidence_artifact_hashes=(_digest("hard-dominated"),),
        ),
        DevelopmentArm(
            arm_id="hard-pareto",
            role=ArmRole.HARD_EXTENSION,
            required_policy_ids=NORMAL_POLICIES,
            hard_extension_policy_ids=("stage_c_repeated_span_template_candidate",),
            exact_natural_tokens=680_000,
            development_gain_lcb_per_token=0.02,
            effect_metric_id="development-risk-reduction-v1",
            effect_metric_artifact_sha256=_digest("metric"),
            common_baseline_artifact_sha256=_digest("common-baseline"),
            all_sensitivity_arms_share_one_common_baseline=True,
            common_baseline_disjoint_from_all_arms=True,
            development_and_heldout_effect_arms_disjoint=True,
            maximum_coverage_js_divergence=0.03,
            minimum_support_recall=1.0,
            extinct_supported_strata=0,
            unknown_mixed_extinct_strata=0,
            representative_linkage_complete=True,
            valid_residuals_complete=True,
            removal_trace_count=200,
            complete_removal_trace_count=200,
            input_manifest_sha256=_digest("input"),
            evidence_artifact_hashes=(_digest("hard-pareto"),),
        ),
    )


def _bundle() -> DevelopmentSelectionBundle:
    protocol = load_development_protocol(ROOT / "configs" / "development_selection_v1.json")
    return DevelopmentSelectionBundle(
        protocol=protocol,
        corpus_slices=_slices(),
        gates=_gates(),
        base_exact_natural_tokens=1_000_000,
        base_input_manifest_sha256=_digest("input"),
        arms=_arms(),
    )


def test_selection_freezes_deterministic_pareto_profiles() -> None:
    # Given a complete disjoint matrix and three eligible candidate arms.
    bundle = _bundle()

    # When development selection is evaluated twice.
    first = select_development_profiles(bundle)
    replay = select_development_profiles(bundle)

    # Then Normal and the nondominated Hard arm are frozen deterministically.
    assert first.status is DevelopmentSelectionStatus.FROZEN
    assert first.normal_arm_id == "normal-candidate"
    assert first.hard_arm_id == "hard-pareto"
    assert first.manifest_sha256 == replay.manifest_sha256
    assert first.benchmark_outcomes_read is False


def test_incomplete_domain_scenario_matrix_is_rejected() -> None:
    # Given one required domain/scenario slice is absent.
    bundle = _bundle()

    # When the boundary parses the incomplete bundle, then admission fails.
    try:
        bundle.model_copy(update={"corpus_slices": bundle.corpus_slices[:-1]}).validate_contract()
    except DevelopmentSelectionContractError as error:
        assert error.reason_code == "development_matrix_incomplete"
    else:
        raise AssertionError("Incomplete development matrix was accepted")


def test_confirmatory_overlap_is_rejected() -> None:
    # Given a development slice overlaps a confirmatory normalized-text hash.
    bundle = _bundle()
    overlapping = bundle.corpus_slices[0].model_copy(update={"confirmatory_text_overlap_count": 1})

    # When disjointness is checked, then the complete bundle is rejected.
    try:
        bundle.model_copy(update={"corpus_slices": (overlapping, *bundle.corpus_slices[1:])}).validate_contract()
    except DevelopmentSelectionContractError as error:
        assert error.reason_code == "development_confirmatory_overlap"
    else:
        raise AssertionError("Overlapping development evidence was accepted")


def test_missing_core_gate_retains_unfrozen_profiles() -> None:
    # Given Quality empirical readiness is missing.
    bundle = _bundle()
    blocked_gates = bundle.gates.model_copy(update={"quality_ready": False})

    # When selection runs, then neither profile is frozen.
    result = select_development_profiles(bundle.model_copy(update={"gates": blocked_gates}))
    assert result.status is DevelopmentSelectionStatus.BLOCKED
    assert result.normal_arm_id is None
    assert result.hard_arm_id is None
    assert "quality_gate_not_ready" in result.blocker_codes


def test_different_sensitivity_baseline_is_rejected() -> None:
    # Given one candidate arm uses a different sensitivity baseline.
    bundle = _bundle()
    drifted = bundle.arms[1].model_copy(update={"common_baseline_artifact_sha256": _digest("other-baseline")})

    # When the effect contract is checked, then common-baseline drift fails.
    try:
        bundle.model_copy(update={"arms": (bundle.arms[0], drifted, bundle.arms[2])}).validate_contract()
    except DevelopmentSelectionContractError as error:
        assert error.reason_code == "development_effect_contract_mismatch"
    else:
        raise AssertionError("Different sensitivity baselines were accepted")


def test_unregistered_hard_extension_is_rejected() -> None:
    # Given a Hard arm names a policy absent from the frozen candidate inventory.
    bundle = _bundle()
    unregistered = bundle.arms[1].model_copy(update={"hard_extension_policy_ids": ("unregistered-policy",)})

    # When the arm is admitted, then it fails before Pareto selection.
    try:
        bundle.model_copy(update={"arms": (bundle.arms[0], unregistered, bundle.arms[2])}).validate_contract()
    except DevelopmentSelectionContractError as error:
        assert error.reason_code == "development_hard_extension_invalid"
    else:
        raise AssertionError("An unregistered Hard extension was accepted")


def test_protocol_loader_rejects_hard_inventory_hash_drift() -> None:
    # Given a copied protocol whose pinned Hard inventory hash is altered.
    protocol = load_development_protocol(ROOT / "configs" / "development_selection_v1.json")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        configs = root / "configs"
        configs.mkdir()
        inventory = ROOT / protocol.hard_candidate_inventory
        (configs / "hard_policy_inventory_v1.json").write_bytes(inventory.read_bytes())
        payload = protocol.model_dump(mode="json")
        payload["hard_candidate_inventory_sha256"] = "0" * 64
        contract = configs / "development_selection_v1.json"
        contract.write_text(json.dumps(payload), encoding="utf-8")

        # When the protocol is loaded, then inventory drift fails closed.
        try:
            load_development_protocol(contract)
        except DevelopmentSelectionContractError as error:
            assert error.reason_code == "development_hard_inventory_hash_mismatch"
        else:
            raise AssertionError("A drifted Hard inventory hash was accepted")


def test_benchmark_feedback_is_not_a_parseable_selector_input() -> None:
    # Given an attempted runtime benchmark-feedback field.
    protocol = load_development_protocol(ROOT / "configs" / "development_selection_v1.json")

    # When Pydantic parses the mutated payload, then the extra field is rejected.
    payload = protocol.model_dump()
    payload["benchmark_results"] = {"humanevalplus": 0.9}
    try:
        type(protocol).model_validate(payload)
    except ValidationError as error:
        assert error.error_count() == 1
    else:
        raise AssertionError("Benchmark feedback entered the development selector contract")


def test_current_repository_preflight_accepts_e2_redundancy_evidence() -> None:
    # Given E1 admission and E2 Redundancy evidence are frozen while Quality/Coverage remain pending.
    report = evaluate_current_development_preflight(ROOT)

    # When readiness is evaluated, then only the unresolved empirical Core gates remain.
    assert report.status is DevelopmentSelectionStatus.BLOCKED
    assert report.profiles_frozen is False
    assert set(report.blocker_codes) == {"quality_gate_not_ready", "coverage_gate_not_ready"}
    assert "redundancy_gate_not_ready" not in report.blocker_codes
    assert "redundancy_gate_evidence_invalid" not in report.blocker_codes
    assert "development_corpus_manifest_not_admitted" not in report.blocker_codes
    assert "development_corpus_manifest_missing" not in report.blocker_codes
    assert "math_benchmark_snapshot_not_frozen" not in report.blocker_codes
    assert "general_benchmark_snapshot_not_frozen" not in report.blocker_codes
    assert not any("benchmark_snapshot_contract_invalid" in blocker for blocker in report.blocker_codes)


if __name__ == "__main__":
    test_selection_freezes_deterministic_pareto_profiles()
    test_incomplete_domain_scenario_matrix_is_rejected()
    test_confirmatory_overlap_is_rejected()
    test_missing_core_gate_retains_unfrozen_profiles()
    test_different_sensitivity_baseline_is_rejected()
    test_unregistered_hard_extension_is_rejected()
    test_protocol_loader_rejects_hard_inventory_hash_drift()
    test_benchmark_feedback_is_not_a_parseable_selector_input()
    test_current_repository_preflight_accepts_e2_redundancy_evidence()
    print("[development-selection-v1] admission, Pareto freeze, and fail-closed preflight: pass")
