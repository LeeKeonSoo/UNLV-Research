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
    assert contract["curation_engine"]["beta_release_enabled_modes"] == ["framework"]
    assert contract["curation_engine"]["production_release_enabled_modes"] == []
    assert contract["stage_a"]["role"] == "validity_hard_gate"
    assert contract["stage_b"]["role"] == (
        "redundancy_removal_and_positive_quality_membership_filtering"
    )
    assert contract["stage_c"]["role"] == "coverage_veto_and_final_materialization"
    assert contract["external_evaluation"]["role"] == "external_offline_validation"
    assert contract["external_evaluation"]["may_mutate_or_reselect_frozen_output"] is False
    assert contract["external_evaluation"]["selector_visible"] is False


def test_stage_b_proposes_and_stage_c_only_vetoes_without_a_fixed_fraction() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["stage_b"]["implemented_decisions"] == [
        "stable_family_nonrepresentative_proposal",
        "reason_coded_qualified_quality_failure_proposal",
        "conjunctive_quality_positive_support_retention",
        "completed_fallback_without_conjunctive_support_non_selection",
    ]
    assert contract["stage_b"]["proposal_is_final_membership"] is False
    assert contract["stage_c"]["may_create_new_removal"] is False
    assert contract["stage_c"]["may_rank_or_impose_quota"] is False
    assert contract["stage_c"]["complete_recheck_required_after_veto"] is True
    assert contract["curation_engine"]["fixed_retention_fraction_allowed"] is False
    assert contract["curation_engine"]["maximum_token_budget_allowed"] is False


def test_semantic_coverage_is_bound_to_stage_c_without_deletion_authority() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    stage_c = contract["stage_c"]
    assert stage_c["semantic_candidate_contract"] == "configs/semantic_coverage_v3.json"
    assert stage_c["allowed_effects"] == [
        "accept_stage_b_non_selection",
        "veto_unexplained_support_loss",
    ]
    assert stage_c["required_retain_is_explicit"] is True
    assert stage_c["silent_restore_allowed"] is False


def test_each_core_has_a_bounded_operational_authority() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    authority = contract["core_authority"]
    assert contract["canonical_cores"] == ["validity", "redundancy", "quality", "coverage"]
    assert authority["validity"]["stage"] == "Stage A"
    assert authority["redundancy"]["stage"] == "Stage B"
    assert authority["quality"]["stage"] == "Stage B"
    assert authority["coverage"]["stage"] == "Stage C"
    assert "unversioned_human_quality_judgment" in contract["runtime_forbidden_inputs"]
    assert contract["composition_artifacts"]["may_remove_or_select"] is False
    assert contract["composition_artifacts"]["comparable_baseline_stage"] == "Stage B pass"
    assert contract["composition_artifacts"]["comparison_unit"] == "chunk"
    assert (
        contract["composition_artifacts"]["raw_record_to_chunk_delta_authority"]
        == "descriptive_only_cross_unit"
    )


def test_contract_does_not_embed_historical_experiment_policy() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert "current_5m_artifact" not in contract
    assert "legacy_nomenclature" not in contract


if __name__ == "__main__":
    test_curation_engine_ends_after_stage_c()
    test_stage_b_proposes_and_stage_c_only_vetoes_without_a_fixed_fraction()
    test_semantic_coverage_is_bound_to_stage_c_without_deletion_authority()
    test_each_core_has_a_bounded_operational_authority()
    test_contract_does_not_embed_historical_experiment_policy()
    print("[operational-curation-v3] Stage A/B/C and external-evaluation boundary: pass")
