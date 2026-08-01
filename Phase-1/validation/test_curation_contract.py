#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "curation_contract.json"


def test_curation_engine_ends_after_stage_c() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["curation_engine"]["included_stages"] == ["Stage A", "Stage B", "Stage C"]
    assert contract["curation_engine"]["curated_output"] == "full_reason_coded_curated_pool"
    assert contract["curation_engine"]["active_policy_profile"] == "normal_structural_v1"
    assert contract["curation_engine"]["user_facing_modes"] == {
        "active": ["normal"],
        "development_only": ["hard"],
        "blocked_for_production": ["hard"],
    }
    assert contract["stage_a"]["role"] == "source_agnostic_text_normalization_and_integrity_handling"
    assert contract["stage_b"]["role"] == "chunk_level_hard_gate"
    assert contract["stage_c"]["role"] == "reason_coded_redundancy_and_quality_retention_without_implicit_budget"
    assert contract["external_evaluation"]["role"] == "external_offline_validation"
    assert contract["external_evaluation"]["may_mutate_or_reselect_frozen_output"] is False
    assert contract["external_evaluation"]["utility_may_enter_stage_c"] is False


def test_stage_c_owns_selection_without_a_fixed_fraction() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["stage_c"]["implemented_decisions"] == [
        "redundancy_compaction",
        "quality_retention_eligibility",
        "reason_coded_materialization",
    ]
    assert contract["stage_c"]["selection_without_binding_budget"] is True
    assert contract["stage_c"]["forbids_implicit_fixed_fraction"] is True
    assert contract["stage_c"]["forbids_utility_and_benchmark_inputs"] is True
    assert contract["stage_c"]["forbids_intrinsic_quality_score"] is True
    assert contract["stage_c"]["quality_retention_contract"] == "quality_retention_deletion_authority_v2"
    assert {
        "policy_scope_route",
        "routing_precondition",
        "policy_id",
        "policy_version",
        "reason_code",
        "observed_evidence",
        "representative_fixture_id",
        "false_positive_fixture_id",
        "original_text_sha256",
        "policy_artifact_sha256",
        "token_delta_proxy",
    } <= set(contract["stage_c"]["quality_reject_required_trace"])
    assert contract["stage_c"]["allowed_policy_inputs"] == ["chunk text", "source-record grouping"]
    assert "language-specific parser" in contract["stage_a"]["language_specific_candidate_boundary"]
    assert "never infer" in contract["stage_a"]["language_specific_candidate_boundary"]


def test_repeated_span_candidate_can_only_operate_after_stage_b_and_before_stage_c_selection() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["stage_c"]["candidate_policies"]["stage_c_repeated_span_template_candidate"]

    assert candidate["status"] == "development_runtime_only_pending_n4_ablation"
    assert candidate["input"] == "stage_b_pass_chunks"
    assert candidate["execution_order"] == ["Stage B hard-gate pass", "candidate span materialization", "active Stage C selection"]
    assert candidate["post_transform_boundary"] == "retain the transformed chunk only when its residual text still meets the declared payload threshold"
    assert candidate["runtime_may_call_candidate"] == "development_only"
    assert "span_occurrence_index" in candidate["required_trace"]


def test_each_core_has_a_bounded_operational_authority() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    authority = contract["core_authority"]
    assert contract["canonical_cores"] == ["validity", "redundancy", "coverage", "quality"]
    assert authority["validity"]["stage"] == "Stage A/B"
    assert authority["redundancy"]["stage"] == "Stage B/C"
    assert authority["quality"]["authority"].startswith("evaluate_retention_eligibility")
    assert authority["coverage"]["authority"].startswith("verify_representative_linkage")
    assert authority["coverage"]["stage"] == "Stage C materialization"
    assert "record_selection" in authority["coverage"]["forbids"]
    assert "quota_based_restoration" in authority["coverage"]["forbids"]
    assert "metadata_strata" in authority["coverage"]["forbids"]
    assert "intrinsic_quality_score" in authority["quality"]["forbids"]
    assert "weighted_priority_score" in authority["quality"]["forbids"]
    assert "human_quality_label" in contract["runtime_forbidden_inputs"]
    assert contract["optional_audit_sidecar"]["may_remove_or_select"] is False


def test_contract_does_not_embed_historical_experiment_policy() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert "current_5m_artifact" not in contract
    assert "legacy_nomenclature" not in contract


if __name__ == "__main__":
    test_curation_engine_ends_after_stage_c()
    test_stage_c_owns_selection_without_a_fixed_fraction()
    test_repeated_span_candidate_can_only_operate_after_stage_b_and_before_stage_c_selection()
    test_each_core_has_a_bounded_operational_authority()
    test_contract_does_not_embed_historical_experiment_policy()
    print("[operational-curation-v3] Stage A/B/C and external-evaluation boundary: pass")
