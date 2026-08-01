#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_curation import resolve_curation_mode


ROOT = Path(__file__).resolve().parents[1]


def test_normal_mode_is_the_only_currently_runnable_user_facing_mode() -> None:
    # Given: the product profile declarations.
    profiles = json.loads((ROOT / "configs" / "policy_profiles.json").read_text(encoding="utf-8"))
    by_id = {profile["id"]: profile for profile in profiles["profiles"]}

    # When: Normal is requested.
    mode = resolve_curation_mode("normal")

    # Then: it resolves to the active structural profile without selector inputs.
    assert mode == {"mode": "normal", "profile_id": "normal_structural_v1"}
    assert by_id[mode["profile_id"]]["status"] == "active"
    assert by_id[mode["profile_id"]]["selector"]["kind"] == "reason_coded_text_structural_only"


def test_hard_mode_fails_closed_until_its_structural_rule_set_is_validated() -> None:
    # Given: a user requests the future stronger profile.
    # When / Then: the runtime rejects it rather than silently applying Normal.
    try:
        resolve_curation_mode("hard")
    except RuntimeError as error:
        assert "Hard" in str(error)
        assert "production use remains blocked" in str(error)
    else:
        raise AssertionError("Hard must fail closed before its policy set is validated")


if __name__ == "__main__":
    test_normal_mode_is_the_only_currently_runnable_user_facing_mode()
    test_hard_mode_fails_closed_until_its_structural_rule_set_is_validated()
    print("[normal-hard-mode-contract] Normal active, Hard production fail-closed: pass")
