#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "content_routing_quality_coverage_contract_v2.json"
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def load_contract() -> dict[str, JsonValue]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_content_router_is_shared_metadata_not_a_core() -> None:
    contract = load_contract()
    router = contract["content_router"]

    assert contract["canonical_cores"] == ["validity", "redundancy", "coverage", "quality"]
    assert router["is_core"] is False
    assert router["authority"] == "shared_observable_metadata_only"
    assert router["consumers"] == ["quality", "coverage"]
    assert router["may_select_or_remove"] is False
    assert router["may_assign_importance"] is False


def test_router_uses_observable_multilabel_axes() -> None:
    router = load_contract()["content_router"]

    assert router["classification"] == "multi_label_with_mixed_unknown_and_ood"
    assert router["axes"] == [
        "content_format",
        "structural_state",
        "language_script",
        "semantic_domain",
    ]
    assert router["quality_routing_priority"] == [
        "content_format",
        "structural_state",
        "language_script",
        "semantic_domain",
    ]
    assert router["training_stage_or_objective_inferred"] is False


def test_quality_is_route_conditioned_but_labels_never_authorize_removal() -> None:
    quality = load_contract()["quality"]

    assert quality["definition"] == "route_conditioned_positive_retention_eligibility"
    assert quality["required_evidence"] == [
        "substantive_payload",
        "route_specific_evidence",
    ]
    assert quality["routing_precondition"]["name"] == "route_confidence"
    assert quality["routing_precondition"]["may_authorize_removal"] is False
    assert "coherence_completeness" not in quality["required_evidence"]
    assert quality["route_label_is_quality_evidence"] is False
    assert quality["domain_membership_may_authorize_removal"] is False
    assert quality["unknown_mixed_or_ood_action"] == "common_rules_only_then_abstain_retain"
    assert quality["global_weighted_score_allowed"] is False
    assert quality["candidate_evidence_registry"] == "configs/route_quality_evidence_candidates_v1.json"
    assert set(quality["registered_route_evidence_status"]) == {
        "general_prose", "code_artifact", "mathematical_content",
        "technical_documentation", "conversation", "instruction", "table_structured_data",
    }


def test_coverage_only_audits_router_labels() -> None:
    coverage = load_contract()["coverage"]

    assert coverage["authority"] == "representation_and_loss_audit_only"
    assert coverage["consumes_shared_router_output"] is True
    assert coverage["may_select_or_remove"] is False
    assert coverage["may_enforce_target_mix_or_quota"] is False
    assert coverage["may_change_quality_threshold"] is False
    assert coverage["may_rescue_quality_reject"] is False


def test_structural_coherence_is_owned_by_validity() -> None:
    contract = load_contract()
    validity = contract["validity"]

    assert validity["owns_structural_coherence_and_integrity"] is True
    assert validity["actions"] == ["repair", "rechunk", "quarantine", "reject"]
    assert validity["original_text_preserved_for_repair_or_rechunk"] is True
    assert contract["quality"]["owns_structural_coherence_and_integrity"] is False


def test_design_is_frozen_without_changing_the_active_runtime() -> None:
    contract = load_contract()

    assert contract["schema_version"] == "content-routing-quality-coverage-contract-v2"
    assert contract["status"] == "design_frozen_not_runtime_active"
    assert contract["runtime_activation"] is False
    assert contract["current_runtime_contract"] == "configs/curation_contract.json"
    assert contract["activation_rule"] == "atomic_contract_runtime_tests_and_docs_switch_after_all_gates_pass"


if __name__ == "__main__":
    test_content_router_is_shared_metadata_not_a_core()
    test_router_uses_observable_multilabel_axes()
    test_quality_is_route_conditioned_but_labels_never_authorize_removal()
    test_coverage_only_audits_router_labels()
    test_structural_coherence_is_owned_by_validity()
    test_design_is_frozen_without_changing_the_active_runtime()
    print("[content-routing-quality-coverage-v2] frozen responsibility boundary: pass")
