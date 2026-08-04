#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_curation import resolve_curation_mode


ROOT = Path(__file__).resolve().parents[1]


def test_normal_mode_is_available_only_as_a_development_candidate() -> None:
    # Given: the product profile declarations.
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    by_id = {profile["id"]: profile for profile in profiles["profiles"]}

    # When: Normal is requested.
    mode = resolve_curation_mode("normal", execution_scope="development")

    # Then: it resolves to the active structural profile without selector inputs.
    assert mode["mode"] == "normal"
    assert mode["profile_id"] == "normal_structural_v1"
    assert mode["authorization"] == "development_candidate_release_blocked"
    assert mode["effective_policy_sha256"]
    assert by_id[mode["profile_id"]]["status"] == "confirmatory_candidate_release_blocked"
    assert by_id[mode["profile_id"]]["selector"]["kind"] == "reason_coded_text_structural_only"


def test_hard_mode_fails_closed_until_its_structural_rule_set_is_validated() -> None:
    # Given: a user requests the future stronger profile.
    # When / Then: the runtime rejects it rather than silently applying Normal.
    try:
        resolve_curation_mode("hard")
    except RuntimeError as error:
        assert "Normal and Hard" in str(error)
        assert "production release is blocked" in str(error)
    else:
        raise AssertionError("Hard must fail closed before its policy set is validated")


if __name__ == "__main__":
    test_normal_mode_is_available_only_as_a_development_candidate()
    test_hard_mode_fails_closed_until_its_structural_rule_set_is_validated()
    print("[normal-hard-mode-contract] candidate modes production fail-closed: pass")
