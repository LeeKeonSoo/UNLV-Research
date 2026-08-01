#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CARD_CONTRACT = ROOT / "configs" / "policy_card_contract.json"
PROFILES = ROOT / "configs" / "policy_profiles.json"


def test_policy_profile_contract() -> None:
    contract = json.loads(CARD_CONTRACT.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "policy-card-contract-v1"
    assert set(contract["required_policy_card_fields"]) >= {
        "id",
        "version",
        "core",
        "stage",
        "hypothesis",
        "allowed_inputs",
        "forbidden_inputs",
        "reason_codes",
        "negative_conditions",
        "coverage_impact",
        "runtime_implementation",
        "empirical_status",
    }
    assert profiles["schema_version"] == "policy-profiles-v1"

    by_id = {profile["id"]: profile for profile in profiles["profiles"]}
    safe = by_id["safe_structural_v3"]
    assert safe["status"] == "historical_frozen"
    assert safe["selector"]["kind"] == "reason_coded_only"
    assert safe["selector"]["reads_utility"] is False
    assert safe["selector"]["reads_benchmark_outcomes"] is False
    assert safe["forbids_implicit_fixed_fraction"] is True

    normal = by_id["normal_structural_v1"]
    assert normal["status"] == "active"
    assert normal["user_facing_mode"] == "normal"
    assert normal["stage_a_policy"] == "text_only_v2"
    assert normal["selector"]["kind"] == "reason_coded_text_structural_only"
    assert normal["selector"]["reads_source_identity"] is False
    assert normal["selector"]["reads_source_tier"] is False
    assert normal["selector"]["reads_rights"] is False
    assert normal["selector"]["reads_path"] is False
    assert normal["runtime_policy"]["stage_a"] == {
        "policy": "text_only_v2",
        "normalization_context_default": "preserve",
    }
    assert normal["runtime_policy"]["stage_b"]["deduplicate_stage_a_text_exactly"] is True
    assert normal["runtime_policy"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is True
    assert normal["runtime_policy"]["stage_c_selection"]["structural_scaffold_compaction"]["enabled"] is True
    assert normal["runtime_policy"]["stage_c_selection"]["structural_artifact_rules"] == {
        "explicit_generated_artifact": True,
        "license_comment_only_chunk": True,
        "empty_html_shell": True,
        "web_chrome_only_chunk": True,
    }
    assert normal["runtime_policy"]["coverage"]["enforce_materialization_invariants"] is True

    hard = by_id["hard_structural_v1"]
    assert hard["status"] == "development_only_pending_n4_ablation"
    assert hard["user_facing_mode"] == "hard"
    assert hard["selector"]["kind"] == "development_only_reason_coded_structural_span_compaction"
    assert hard["forbids_implicit_fixed_fraction"] is True

    calibrated = by_id["calibrated_selector_template_v1"]
    assert calibrated["status"] == "retired_not_runnable"
    assert calibrated["selector"]["kind"] == "retired_score_selector"
    assert calibrated["selector"]["activation_requirements"] == [
        "frozen_reference_data",
        "declared_score_direction",
        "held_out_calibration",
        "executable_negative_condition_test",
        "cross_domain_or_scope_audit",
        "external_validation_without_benchmark_feedback",
    ]
    assert calibrated["selector"]["reads_utility"] is False
    assert calibrated["selector"]["reads_benchmark_outcomes"] is False

    aggressive = by_id["aggressive_structural_candidate_v1"]
    assert aggressive["status"] == "archived_candidate_research"
    assert aggressive["selector"]["kind"] == "candidate_reason_coded_text_structural_ablation"
    assert aggressive["selector"]["reads_source_identity"] is False
    assert aggressive["selector"]["reads_benchmark_outcomes"] is False
    assert aggressive["forbids_implicit_fixed_fraction"] is True

    quality_candidate = by_id["quality_retention_candidate_v1"]
    assert quality_candidate["status"] == "development_only"
    assert quality_candidate["inherits_profile"] == "normal_structural_v1"
    assert quality_candidate["selector"]["kind"] == "development_only_quality_retention_evidence"
    assert quality_candidate["selector"]["reads_benchmark_outcomes"] is False

    from run_curation import resolve_curation_mode, validate_run_policy_overrides

    resolved = resolve_curation_mode("normal")
    assert resolved["profile_id"] == "normal_structural_v1"
    assert len(resolved["effective_policy_sha256"]) == 64
    assert resolved["effective_policy"] == normal["runtime_policy"]
    validate_run_policy_overrides({}, resolved["effective_policy"])
    try:
        validate_run_policy_overrides(
            {"stage_c_selection": {"near_duplicate_compaction": {"candidate_enabled": True}}},
            resolved["effective_policy"],
        )
    except RuntimeError as error:
        assert "immutable profile" in str(error)
    else:
        raise AssertionError("A run contract must not override an immutable user-facing profile.")


if __name__ == "__main__":
    test_policy_profile_contract()
    print("[policy-profile-contract] safe and calibrated-selector boundaries: pass")
