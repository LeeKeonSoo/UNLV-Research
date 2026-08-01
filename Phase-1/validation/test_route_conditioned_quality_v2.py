#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from route_conditioned_quality import (
    EvidenceContractError,
    EvidenceHead,
    QualityUnit,
    RouteEvidenceBundle,
    evaluate_route_conditioned_quality,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FIXTURES = ROOT / "validation" / "fixtures" / "route_conditioned_quality_v2_cases.json"
REGISTRY = ROOT / "configs" / "route_conditioned_quality_v2.json"
CORE_REGISTRY = ROOT / "configs" / "core_policy_registry.json"
POLICY_CARDS = ROOT / "configs" / "policy_cards.json"
ARTIFACT_HASH = "a" * 64


def load_cases() -> list[dict[str, JsonValue]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list)
    return cases


def head(name: str, outcome: str) -> EvidenceHead:
    negative_boundary = f"fixture_{name}_negative_v1" if outcome == "negative" else None
    return EvidenceHead(
        name=name,
        outcome=outcome,
        evidence_id=f"fixture_{name}_v1",
        provider_version="fixture-v1",
        artifact_sha256=ARTIFACT_HASH,
        negative_boundary_id=negative_boundary,
    )


def bundle(raw: dict[str, JsonValue]) -> RouteEvidenceBundle:
    return RouteEvidenceBundle(
        route=raw["route"],
        substantive_payload=head("substantive_payload", raw["substantive_payload"]),
        route_specific_evidence=head("route_specific_evidence", raw["route_specific_evidence"]),
        profile_id="fixture-profile-v1",
        profile_sha256=ARTIFACT_HASH,
    )


def test_fixture_matrix_enforces_routing_precondition_and_two_quality_heads() -> None:
    for case in load_cases():
        text = case["text"]
        raw_bundles = case["bundles"]
        assert isinstance(text, str)
        assert isinstance(raw_bundles, list)
        result = evaluate_route_conditioned_quality(
            QualityUnit(text=text, evidence_bundles=tuple(bundle(raw) for raw in raw_bundles))
        )

        assert result.decision == case["expected_decision"], case["id"]
        assert result.reason_code == case["expected_reason_code"], case["id"]
        assert result.authority == "candidate_quality_only"
        assert result.may_mutate_curated_membership is False
        assert result.routing_precondition == (
            "pass" if result.route_status == "routed" else "indeterminate"
        )
        assert not hasattr(result, "route_confidence_head")


def test_negative_outcome_requires_a_named_boundary() -> None:
    try:
        EvidenceHead(
            name="substantive_payload",
            outcome="negative",
            evidence_id="invalid-negative",
            provider_version="fixture-v1",
            artifact_sha256=ARTIFACT_HASH,
            negative_boundary_id=None,
        )
    except EvidenceContractError as error:
        assert "negative boundary" in str(error)
    else:
        raise AssertionError("An unnamed negative boundary must not authorize rejection.")


def test_registry_separates_routing_precondition_from_two_quality_heads() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["status"] == "candidate_not_runtime_active"
    assert registry["required_evidence_heads"] == [
        "substantive_payload",
        "route_specific_evidence",
    ]
    assert registry["routing_precondition"]["name"] == "route_confidence"
    assert registry["routing_precondition"]["quality_evidence"] is False
    assert "coherence_completeness" not in registry["required_evidence_heads"]
    assert registry["global_weighted_score_allowed"] is False
    assert registry["route_label_is_quality_evidence"] is False
    assert registry["unknown_mixed_ood_action"] == "abstain_retain"


def test_candidate_is_linked_to_quality_core_without_runtime_authority() -> None:
    core_registry = json.loads(CORE_REGISTRY.read_text(encoding="utf-8"))
    policy_cards = json.loads(POLICY_CARDS.read_text(encoding="utf-8"))
    policies = {policy["id"]: policy for policy in core_registry["policies"]}
    cards = {card["id"]: card for card in policy_cards["cards"]}

    candidate = policies["stage_c_route_conditioned_quality_candidate"]
    assert candidate["core"] == "quality"
    assert candidate["status"] == "candidate"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert cards[candidate["policy_card_id"]]["empirical_status"] == (
        "candidate_adapter_validated_all_routes_abstain_missing_or_indeterminate_evidence_not_runtime_active"
    )
    assert cards[candidate["policy_card_id"]]["evidence_gate"] == (
        "configs/quality_route_evidence_gate_v2.json"
    )
    assert cards[candidate["policy_card_id"]]["candidate_evidence_registry"] == (
        "configs/route_quality_evidence_candidates_v1.json"
    )
    assert "route_quality_evidence_candidates.py" in candidate["runtime_implementation"]


if __name__ == "__main__":
    test_fixture_matrix_enforces_routing_precondition_and_two_quality_heads()
    test_negative_outcome_requires_a_named_boundary()
    test_registry_separates_routing_precondition_from_two_quality_heads()
    test_candidate_is_linked_to_quality_core_without_runtime_authority()
    print("[route-conditioned-quality-v2] routing precondition and two-head boundary: pass")
