#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_profiles import (
    ProfileContractError,
    ProfileRegistry,
    load_profile_registry,
    validate_retained_set_monotonicity,
)

MANIFEST = ROOT / "configs" / "curation_framework_v1.json"
OBJECTS = ROOT / "configs" / "framework_objects_v1.json"
PROFILES = ROOT / "configs" / "framework_profiles_v1.json"


def test_only_normal_and_hard_profiles_are_public() -> None:
    # Given: the redesigned profile registry.
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    by_id = {profile.id.value: profile for profile in registry.profiles}

    # When / Then: the product surface contains exactly two budget-free modes.
    assert set(by_id) == {"normal", "hard"}
    assert all(not profile.fixed_retention_fraction_allowed for profile in registry.profiles)
    assert all(not profile.maximum_token_budget_allowed for profile in registry.profiles)


def test_normal_and_hard_share_policy_families_with_distinct_operating_points() -> None:
    # Given: Normal and Hard composition.
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    by_id = {profile.id.value: profile for profile in registry.profiles}
    normal = by_id["normal"]
    hard = by_id["hard"]

    # When / Then: both modes apply the same Policies through separately calibrated strengths.
    assert normal.policy_ids == hard.policy_ids
    assert hard.inherits_profile == "normal"
    assert set(normal.policy_ids) == {
        "validity.interpretable_text",
        "redundancy.exact_text_family",
        "redundancy.symmetric_near_duplicate_candidate",
        "quality.explicit_nonpayload",
        "coverage.representative_guard",
        "quality.distilled_ranker_v1",
    }
    assert normal.operating_point_id == "normal_v1"
    assert hard.operating_point_id == "hard_v1"
    assert normal.strength_rank == 1
    assert hard.strength_rank == 2
    assert normal.calibration_artifact_sha256 is None
    assert hard.calibration_artifact_sha256 is None


def test_profiles_remain_blocked_until_all_policies_are_promoted() -> None:
    # Given: design-only policies mapped from the current runtime and E3 evidence.
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)

    # When / Then: composition exists without pretending it is release-ready.
    assert all(not profile.release_enabled for profile in registry.profiles)
    assert "profile_contains_unpromoted_policy" in registry.blocker_codes


def test_release_enabled_profile_cannot_reference_unpromoted_policy() -> None:
    # Given: the design registry with Normal incorrectly marked releasable.
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    payload = registry.model_dump(mode="json")
    normal = next(profile for profile in payload["profiles"] if profile["id"] == "normal")
    normal["release_enabled"] = True
    normal["calibration_artifact_sha256"] = "f" * 64

    # When / Then: lifecycle state prevents accidental activation.
    try:
        ProfileRegistry.model_validate(payload)
    except ValidationError as error:
        assert "profile_release_contains_unpromoted_policy" in str(error)
    else:
        raise AssertionError("Unpromoted Policies entered a released profile")


def test_materialized_hard_set_must_be_subset_of_normal() -> None:
    # Given: valid and invalid materialized retained-ID sets.
    validate_retained_set_monotonicity(
        normal_retained=("a", "b", "c"),
        hard_retained=("a", "c"),
    )

    # When / Then: a Hard-only survivor is a release-blocking error.
    try:
        validate_retained_set_monotonicity(
            normal_retained=("a", "b"),
            hard_retained=("a", "c"),
        )
    except ProfileContractError as error:
        assert error.reason_code == "profile_hard_retained_set_not_subset"
        assert error.offending_ids == ("c",)
    else:
        raise AssertionError("Hard materialization violated subset monotonicity")


if __name__ == "__main__":
    test_only_normal_and_hard_profiles_are_public()
    test_normal_and_hard_share_policy_families_with_distinct_operating_points()
    test_profiles_remain_blocked_until_all_policies_are_promoted()
    test_release_enabled_profile_cannot_reference_unpromoted_policy()
    test_materialized_hard_set_must_be_subset_of_normal()
    print("[framework-profiles-v1] Normal/Hard composition and monotonicity: pass")
