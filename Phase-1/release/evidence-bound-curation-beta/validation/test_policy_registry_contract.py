#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "runtime_policy_registry_v1.json"
PROFILES = ROOT / "configs" / "runtime_policy_profiles_v1.json"
DEPLOYMENT = ROOT / "configs" / "deployment_surface_v1.json"


def test_runtime_registry_is_exactly_the_public_policy_surface() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    policies = {policy["id"]: policy for policy in registry["policies"]}
    expected = set(deployment["active_policy_ids"])

    assert registry["schema_version"] == "runtime-policy-registry-v1"
    assert registry["policy_count"] == len(policies) == 13
    assert set(policies) == expected
    assert {profile["user_facing_mode"] for profile in profiles["profiles"]} == {
        "framework",
    }
    assert all(set(profile["enabled_policy_ids"]) == expected for profile in profiles["profiles"])


def test_every_policy_resolves_implementation_and_fixture_files() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for policy in registry["policies"]:
        assert policy["core"] in {"validity", "redundancy", "quality", "coverage"}
        assert policy["stage"] in {"stage_a", "stage_b", "stage_c"}
        assert policy["metric"].startswith(policy["core"] + ".")
        assert "." in policy["method"]
        assert policy["runtime_implementation"]
        assert all((ROOT / path).is_file() for path in policy["runtime_implementation"])
        assert (ROOT / policy["positive_fixture"]).is_file()
        assert (ROOT / policy["false_positive_fixture"]).is_file()


if __name__ == "__main__":
    test_runtime_registry_is_exactly_the_public_policy_surface()
    test_every_policy_resolves_implementation_and_fixture_files()
    print("[runtime-policy-registry-v1] exact policy, method, and fixture surface: pass")
