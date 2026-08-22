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
    assert normal["status"] == "confirmatory_candidate_release_blocked"
    assert normal["user_facing_mode"] == "normal"
    assert normal["stage_a_policy"] == "text_only_v2"
    assert normal["selector"]["kind"] == "reason_coded_structural_and_distilled_quality_evidence"
    assert normal["selector"]["reads_source_identity"] is False
    assert normal["selector"]["reads_source_tier"] is False
    assert normal["selector"]["reads_rights"] is False
    assert normal["selector"]["reads_path"] is False
    assert normal["runtime_policy"]["stage_a"] == {
        "policy": "text_only_v2",
        "normalization_context_default": "preserve",
    }
    assert normal["runtime_policy"]["stage_b"]["deduplicate_stage_a_text_exactly"] is True
    assert normal["runtime_policy"]["redundancy_v2"]["runtime_activation"] is True
    assert normal["runtime_policy"]["quality_runtime"]["runtime_required"] is True
    assert normal["runtime_policy"]["quality_runtime"]["minimum_passed_policies"] == 1
    assert normal["runtime_policy"]["quality_runtime"]["abstain_action"] == (
        "not_select_unless_coverage_veto"
    )
    assert "stage_b_quality_distilled_ranker_v1" in normal["enabled_policy_ids"]
    assert "stage_b_intra_chunk_exact_sentence_compaction" in normal[
        "enabled_policy_ids"
    ]
    assert "near_duplicate_compaction" not in normal["runtime_policy"]["stage_b_policy"]
    assert normal["runtime_policy"]["stage_b_policy"][
        "intra_chunk_exact_sentence_compaction"
    ] == {
        "enabled": True,
        "minimum_occurrences": 3,
        "minimum_lexical_tokens": 12,
    }
    assert normal["runtime_policy"]["stage_b_policy"]["structural_scaffold_compaction"]["enabled"] is True
    assert normal["runtime_policy"]["stage_b_policy"]["structural_artifact_rules"] == {
        "explicit_generated_artifact": True,
        "license_comment_only_chunk": True,
        "empty_html_shell": True,
        "web_chrome_only_chunk": True,
    }
    assert normal["runtime_policy"]["coverage"]["enforce_materialization_invariants"] is True

    hard = by_id["hard_structural_v1"]
    assert hard["status"] == "confirmatory_candidate_release_blocked"
    assert hard["user_facing_mode"] == "hard"
    assert hard["selector"]["kind"] == "reason_coded_structural_and_distilled_quality_evidence"
    assert hard["forbids_implicit_fixed_fraction"] is True
    assert hard["runtime_policy"]["redundancy_v2"]["runtime_activation"] is True
    assert hard["runtime_policy"]["quality_runtime"]["runtime_required"] is True
    assert hard["runtime_policy"]["quality_runtime"]["minimum_passed_policies"] == 2
    assert hard["enabled_policy_ids"] == normal["enabled_policy_ids"]
    assert "hard_span_compaction" not in hard["runtime_policy"]["stage_b_policy"]
    assert set(hard["candidate_policy_ids"]) == {
        "stage_b_inline_license_header_candidate",
        "stage_b_inline_license_comment_block_candidate",
        "stage_b_repeated_span_template_candidate",
    }

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

    runtime_profiles = json.loads(
        (ROOT / "configs" / "runtime_policy_profiles_v1.json").read_text(
            encoding="utf-8"
        )
    )
    framework = runtime_profiles["profiles"][0]
    resolved = resolve_curation_mode("framework", execution_scope="development")
    assert resolved["profile_id"] == "framework_structural_v2"
    assert len(resolved["effective_policy_sha256"]) == 64
    assert resolved["effective_policy"] == framework["runtime_policy"]
    validate_run_policy_overrides({}, resolved["effective_policy"])
    try:
        validate_run_policy_overrides(
            {"stage_b_policy": {"near_duplicate_compaction": {"candidate_enabled": True}}},
            resolved["effective_policy"],
        )
    except RuntimeError as error:
        assert "immutable profile" in str(error)
    else:
        raise AssertionError("A run contract must not override an immutable user-facing profile.")


if __name__ == "__main__":
    test_policy_profile_contract()
    print("[policy-profile-contract] safe and calibrated-selector boundaries: pass")
