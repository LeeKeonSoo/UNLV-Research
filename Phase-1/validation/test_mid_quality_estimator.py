#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mid_quality_estimator import build_mid_quality_development_report


def test_mid_quality_estimator_removes_only_groups_with_calibrated_non_positive_upper_bound() -> None:
    # Given: repeated heldout-loss ablations and a frozen null-control calibration set.
    groups = [
        {"group_id": "explicit-web-chrome", "effect_samples": [-0.060, -0.050, -0.055]},
        {"group_id": "uncertain-template", "effect_samples": [-0.030, 0.010, -0.010]},
        {"group_id": "useful-example", "effect_samples": [0.040, 0.050, 0.045]},
    ]

    # When: candidate-only Mid development evidence is summarized.
    report = build_mid_quality_development_report(
        groups=groups,
        null_control_effect_samples=[-0.003, 0.002, 0.004, -0.002, 0.001],
        confidence_level=0.95,
        bootstrap_replicates=400,
        random_seed=101,
    )

    # Then: only a confidently non-positive marginal contribution is removable.
    decisions = {row["group_id"]: row for row in report["groups"]}
    assert decisions["explicit-web-chrome"]["decision"] == "candidate_remove"
    assert decisions["explicit-web-chrome"]["upper_confidence_bound"] <= 0.0
    assert decisions["uncertain-template"]["decision"] == "candidate_retain_uncertain"
    assert decisions["useful-example"]["decision"] == "candidate_retain_positive"
    assert report["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert report["calibration"]["method"] == "null_control_bootstrap_margin"


def test_mid_quality_estimator_rejects_forbidden_runtime_inputs() -> None:
    # Given: a group record contaminated with a prohibited runtime signal.
    groups = [{"group_id": "bad", "effect_samples": [-0.1, -0.1], "benchmark_outcomes": 1.0}]

    # When / Then: the development boundary rejects the record.
    try:
        build_mid_quality_development_report(
            groups=groups,
            null_control_effect_samples=[0.0, 0.0],
            confidence_level=0.95,
            bootstrap_replicates=100,
            random_seed=101,
        )
    except RuntimeError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("Forbidden runtime input must be rejected")


if __name__ == "__main__":
    test_mid_quality_estimator_removes_only_groups_with_calibrated_non_positive_upper_bound()
    test_mid_quality_estimator_rejects_forbidden_runtime_inputs()
    print("[mid-quality-estimator] calibrated candidate-only decisions: pass")
