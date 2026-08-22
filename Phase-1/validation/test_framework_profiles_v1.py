#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_profiles import ProfileRegistry, load_profile_registry

MANIFEST = ROOT / "configs" / "curation_framework_v1.json"
OBJECTS = ROOT / "configs" / "framework_objects_v1.json"
PROFILES = ROOT / "configs" / "framework_profiles_v1.json"


def test_only_one_budget_free_framework_profile_is_public() -> None:
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    assert tuple(profile.id.value for profile in registry.profiles) == ("framework",)
    profile = registry.profiles[0]
    assert profile.inherits_profile is None
    assert profile.operating_point_id == "framework_v2"
    assert profile.strength_rank == 1
    assert profile.fixed_retention_fraction_allowed is False
    assert profile.maximum_token_budget_allowed is False
    assert registry.retained_set_invariant == "single_profile"


def test_framework_profile_composes_all_four_core_policy_families() -> None:
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    assert set(registry.profiles[0].policy_ids) == {
        "validity.interpretable_text",
        "redundancy.exact_text_family",
        "redundancy.symmetric_near_duplicate_candidate",
        "redundancy.intra_chunk_exact_sentence_compaction",
        "quality.explicit_nonpayload",
        "coverage.representative_guard",
        "quality.distilled_ranker_v1",
    }


def test_profile_remains_blocked_until_all_policies_are_promoted() -> None:
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    assert registry.profiles[0].release_enabled is False
    assert "profile_contains_unpromoted_policy" in registry.blocker_codes


def test_release_enabled_profile_cannot_reference_unpromoted_policy() -> None:
    registry = load_profile_registry(MANIFEST, OBJECTS, PROFILES)
    payload = registry.model_dump(mode="json")
    payload["profiles"][0]["release_enabled"] = True
    payload["profiles"][0]["calibration_artifact_sha256"] = "f" * 64
    try:
        ProfileRegistry.model_validate(payload)
    except ValidationError as error:
        assert "profile_release_contains_unpromoted_policy" in str(error)
    else:
        raise AssertionError("Unpromoted Policies entered a released profile")


if __name__ == "__main__":
    test_only_one_budget_free_framework_profile_is_public()
    test_framework_profile_composes_all_four_core_policy_families()
    test_profile_remains_blocked_until_all_policies_are_promoted()
    test_release_enabled_profile_cannot_reference_unpromoted_policy()
    print("[framework-profiles-v1] single framework profile composition: pass")
