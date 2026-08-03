#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_qualification import (
    ProtectedFixtureGate,
    QualificationContractError,
    exact_one_sided_upper_bound,
)


def test_zero_of_eight_hundred_supports_normal_false_removal_gate() -> None:
    # Given: a frozen protected set of 800 fixtures with no false removals.
    gate = ProtectedFixtureGate(sample_count=800, false_removal_count=0, maximum_upper_bound=0.005)

    # When: the exact one-sided 95% bound is evaluated.
    upper_bound = gate.upper_bound(confidence=0.95)

    # Then: the Normal 0.5% gate is supported.
    assert upper_bound < 0.005
    assert gate.passes(confidence=0.95)


def test_five_hundred_twelve_total_fixtures_do_not_prove_normal_gate() -> None:
    # Given: the proposed 512-fixture smoke suite has zero observed errors.
    upper_bound = exact_one_sided_upper_bound(successes=0, trials=512, confidence=0.95)

    # When/Then: its uncertainty remains above the Normal 0.5% limit.
    assert upper_bound > 0.005


def test_invalid_binomial_observation_is_rejected() -> None:
    # Given/When/Then: impossible false-removal counts are rejected at the boundary.
    try:
        ProtectedFixtureGate(sample_count=10, false_removal_count=11, maximum_upper_bound=0.02)
    except QualificationContractError as error:
        assert "cannot exceed" in str(error)
    else:
        raise AssertionError("Impossible binomial observations must be rejected")


if __name__ == "__main__":
    test_zero_of_eight_hundred_supports_normal_false_removal_gate()
    test_five_hundred_twelve_total_fixtures_do_not_prove_normal_gate()
    test_invalid_binomial_observation_is_rejected()
    print("[quality-teacher-qualification-v1] uncertainty gates: pass")
