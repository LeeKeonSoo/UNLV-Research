#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from positive_quality_evidence import (
    KNOWN_ROUTES,
    CalibrationManifest,
    CalibrationRow,
    ChunkEvidence,
    EvidenceContractError,
    Route,
    RouteEvidence,
    RouteThresholds,
    ThresholdProfile,
    calibrate_threshold_profiles,
    evaluate_positive_quality,
    wilson_lower_bound,
    wilson_upper_bound,
)


PROVIDER_HASH = "a" * 64


def evidence(route: Route, score: float, uid: str = "chunk") -> ChunkEvidence:
    return ChunkEvidence(
        chunk_uid=uid,
        routes=(RouteEvidence(route, score, score, score, score),),
        provider_manifest_sha256=PROVIDER_HASH,
    )


def profile(profile_id: str, score: float) -> ThresholdProfile:
    return ThresholdProfile(
        profile_id,
        tuple(RouteThresholds(route, score, score, score, score) for route in KNOWN_ROUTES),
    )


def test_all_evidence_heads_must_pass_without_a_weighted_score() -> None:
    thresholds = profile("frozen", 0.8)
    passing = evidence("code", 0.9, "pass")
    partial = ChunkEvidence(
        "partial",
        (RouteEvidence("code", 0.9, 0.9, 0.7, 0.9),),
        PROVIDER_HASH,
    )

    assert evaluate_positive_quality(passing, thresholds).decision == "eligible_keep"
    assert evaluate_positive_quality(partial, thresholds).decision == "abstain"
    assert evaluate_positive_quality(partial, thresholds).qualifying_routes == ()


def test_provider_native_scores_need_not_be_probabilities() -> None:
    thresholds = ThresholdProfile(
        "native-logit-scale",
        (RouteThresholds("general_prose", 1.0, 2.5, 1.5, 2.0),),
    )
    raw_logits = ChunkEvidence(
        "native-logits",
        (RouteEvidence("general_prose", 1.0, 3.2, 2.1, 2.8),),
        PROVIDER_HASH,
    )

    assert evaluate_positive_quality(raw_logits, thresholds).decision == "eligible_keep"


def test_unknown_missing_and_explicit_reject_boundaries() -> None:
    thresholds = profile("frozen", 0.8)
    unknown = evidence("unknown", 1.0, "unknown")
    missing = ChunkEvidence("missing", (), PROVIDER_HASH)
    positive = evidence("math", 0.95, "positive")

    assert evaluate_positive_quality(unknown, thresholds).decision == "abstain"
    assert evaluate_positive_quality(missing, thresholds).decision == "abstain"
    rejected = evaluate_positive_quality(positive, thresholds, "explicit_non_payload_fixture")
    assert rejected.decision == "reject"
    assert rejected.reason_code == "explicit_non_payload_fixture"


def test_calibration_selects_most_compressive_feasible_profile() -> None:
    rows: list[CalibrationRow] = []
    calibration_groups: set[str] = set()
    for route in KNOWN_ROUTES:
        source_group = f"calibration-{route}"
        calibration_groups.add(source_group)
        rows.extend(
            CalibrationRow(
                evidence(route, 0.95, f"{route}-{index}"),
                10,
                "clean_control",
                route,
                source_group,
            )
            for index in range(400)
        )
    rows.extend(
        (
            CalibrationRow(evidence("general_prose", 0.4, "candidate-low"), 100, "candidate_pool", None, "candidate"),
            CalibrationRow(evidence("general_prose", 0.7, "candidate-mid"), 200, "candidate_pool", None, "candidate"),
            CalibrationRow(evidence("general_prose", 0.9, "candidate-high"), 300, "candidate_pool", None, "candidate"),
        )
    )
    manifest = CalibrationManifest(
        frozenset(f"training-{route}" for route in KNOWN_ROUTES),
        frozenset(calibration_groups),
    )
    result = calibrate_threshold_profiles(
        tuple(rows),
        (profile("loose", 0.5), profile("hard", 0.8), profile("over_strict", 0.99)),
        manifest,
        0.01,
    )

    assert result.selected_profile_id == "hard"
    assert result.target_retention_fraction_used is False
    by_id = {report.profile_id: report for report in result.profiles}
    assert by_id["hard"].feasible is True
    assert by_id["hard"].excluded_candidate_tokens == 300
    assert by_id["over_strict"].feasible is False
    assert all(bound <= 0.01 for _, bound in by_id["hard"].route_false_reject_upper_bounds)


def test_calibration_rejects_source_overlap() -> None:
    try:
        CalibrationManifest(frozenset({"shared"}), frozenset({"shared"}))
    except EvidenceContractError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("Overlapping provider-training and calibration sources must fail.")


def test_public_wilson_bound_is_conservative_for_zero_failures() -> None:
    assert 0.0 < wilson_upper_bound(0, 65, 0.95) < 0.05
    assert wilson_upper_bound(1, 65, 0.95) > 0.05


def test_public_wilson_lower_bound_is_dual_to_failure_upper_bound() -> None:
    assert wilson_lower_bound(90, 100, 0.95) == 1.0 - wilson_upper_bound(10, 100, 0.95)


if __name__ == "__main__":
    test_all_evidence_heads_must_pass_without_a_weighted_score()
    test_provider_native_scores_need_not_be_probabilities()
    test_unknown_missing_and_explicit_reject_boundaries()
    test_calibration_selects_most_compressive_feasible_profile()
    test_calibration_rejects_source_overlap()
    test_public_wilson_bound_is_conservative_for_zero_failures()
    test_public_wilson_lower_bound_is_dual_to_failure_upper_bound()
    print("[positive-quality-evidence] conjunctive evidence and calibration: pass")
