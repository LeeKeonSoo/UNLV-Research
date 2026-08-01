#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_qurater_general_prose import (
    CalibrationInputError,
    GeneralProseScore,
    build_threshold_profiles,
    ensure_disjoint,
)


def score(uid: str, digest: str, value: float) -> GeneralProseScore:
    return GeneralProseScore(uid, digest, True, 10, value, 0.0, value, value)


def test_threshold_profiles_use_declared_clean_control_quantiles() -> None:
    clean = (
        GeneralProseScore("out", "e" * 64, False, 10, -100.0, 0.0, -100.0, -100.0),
        score("a", "a" * 64, 1.0),
        score("b", "b" * 64, 2.0),
        score("c", "c" * 64, 3.0),
        score("d", "d" * 64, 4.0),
    )

    profiles = build_threshold_profiles(clean, (0.0, 0.5))

    assert [profile.profile_id for profile in profiles] == ["clean_q0", "clean_q0.5"]
    assert profiles[0].routes[0].substantive_payload == 1.0
    assert profiles[1].routes[0].substantive_payload == 2.0
    assert profiles[1].routes[0].coherence_completeness == 2.0
    assert profiles[1].routes[0].route_specific_evidence == 2.0


def test_clean_and_candidate_ids_and_hashes_must_be_disjoint() -> None:
    clean = (score("clean", "a" * 64, 1.0),)

    for candidate in (
        (score("clean", "b" * 64, 2.0),),
        (score("candidate", "a" * 64, 2.0),),
    ):
        try:
            ensure_disjoint(clean, candidate)
        except CalibrationInputError:
            continue
        raise AssertionError("Calibration pools with overlapping identities must fail.")


if __name__ == "__main__":
    test_threshold_profiles_use_declared_clean_control_quantiles()
    test_clean_and_candidate_ids_and_hashes_must_be_disjoint()
    print("[qurater-general-prose-calibration] quantiles and disjointness: pass")
