#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validity_recovery import ValidityUnit, evaluate_validity


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FIXTURES = ROOT / "validation" / "fixtures" / "validity_recovery_v1_cases.json"
REGISTRY = ROOT / "configs" / "validity_recovery_v1.json"
CORE_REGISTRY = ROOT / "configs" / "core_policy_registry.json"
POLICY_CARDS = ROOT / "configs" / "policy_cards.json"


def load_cases() -> list[dict[str, JsonValue]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list)
    return cases


def test_validity_recovery_cases_follow_the_declared_action_order() -> None:
    for case in load_cases():
        text = case["text"]
        source_text = case.get("source_record_text")
        assert isinstance(text, str)
        assert source_text is None or isinstance(source_text, str)

        result = evaluate_validity(ValidityUnit(text=text, source_record_text=source_text))

        assert result.final_action == case["expected_final_action"], case["id"]
        assert list(result.action_trace) == case["expected_action_trace"], case["id"]
        assert set(case["expected_reason_codes"]) <= set(result.reason_codes), case["id"]
        expected_recovered = case.get("expected_recovered_text", text)
        assert result.recovered_text == expected_recovered, case["id"]
        assert result.original_text == text, case["id"]


def test_repair_and_rechunk_preserve_hash_and_transformation_trace() -> None:
    result = evaluate_validity(
        ValidityUnit(
            text="The bird\u0092s example starts a fence.\n```text",
            source_record_text="The bird\u0092s example starts a fence.\n```text\npayload\n```",
        )
    )

    assert result.final_action == "rechunk"
    assert result.original_sha256 != result.recovered_sha256
    assert result.source_record_sha256 is not None
    assert "cp1252_c1_0092_to_2019" in result.transformation_codes
    assert result.authority == "candidate_validity_only"
    assert result.may_select_for_training is False


def test_registry_defines_reject_as_last_resort() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["status"] == "candidate_not_runtime_active"
    assert registry["action_order"] == ["repair", "rechunk", "quarantine", "reject"]
    assert registry["reject_boundary"] == "no_interpretable_payload_after_declared_recovery"
    assert registry["original_text_preservation_required"] is True
    assert registry["quality_authority"] is False


def test_candidate_is_linked_to_core_registry_and_policy_card() -> None:
    core_registry = json.loads(CORE_REGISTRY.read_text(encoding="utf-8"))
    policy_cards = json.loads(POLICY_CARDS.read_text(encoding="utf-8"))
    policies = {policy["id"]: policy for policy in core_registry["policies"]}
    cards = {card["id"]: card for card in policy_cards["cards"]}

    candidate = policies["stage_a_b_validity_recovery_candidate"]
    card = cards["stage_a_b_validity_recovery_candidate"]
    assert candidate["core"] == "validity"
    assert candidate["status"] == "candidate"
    assert candidate["runtime_authorization"] == "none_candidate_cannot_quarantine_or_reject"
    assert candidate["policy_card_id"] == card["id"]
    assert "validity_recovery.py" in candidate["runtime_implementation"]
    assert card["empirical_status"] == "fixture_validated_candidate_not_runtime_active"


if __name__ == "__main__":
    test_validity_recovery_cases_follow_the_declared_action_order()
    test_repair_and_rechunk_preserve_hash_and_transformation_trace()
    test_registry_defines_reject_as_last_resort()
    test_candidate_is_linked_to_core_registry_and_policy_card()
    print("[validity-recovery-v1] repair/rechunk/quarantine/reject boundary: pass")
