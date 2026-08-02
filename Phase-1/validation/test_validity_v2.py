#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validity_v2 import TextField, ValidityInput, evaluate_validity_v2
from validity_v2_audit import ValidityAuditCase, build_validity_audit


FIXTURES = ROOT / "validation" / "fixtures" / "validity_v2_cases.json"
CONTRACT = ROOT / "configs" / "validity_v2.json"


def load_cases() -> list[dict[str, str | dict[str, str]]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    return payload["cases"]


def _input(case: dict[str, str | dict[str, str]]) -> ValidityInput:
    fields = case["fields"]
    assert isinstance(fields, dict)
    source = case.get("source_record_text")
    assert source is None or isinstance(source, str)
    return ValidityInput(
        text_fields=tuple(TextField(name, text) for name, text in fields.items()),
        source_record_text=source,
    )


def test_cross_domain_cases_follow_the_four_state_contract() -> None:
    for case in load_cases():
        decision = evaluate_validity_v2(_input(case))

        assert decision.status.value == case["expected_status"], case["id"]
        assert decision.action.value == case["expected_action"], case["id"]
        expected_reason = case.get("expected_reason")
        if isinstance(expected_reason, str):
            assert expected_reason in decision.reason_codes, case["id"]
        assert decision.provider_outputs_read is False
        assert decision.benchmark_outcomes_read is False
        assert decision.utility_read is False


def test_raw_byte_boundaries_repair_quarantine_and_reject_without_guessing() -> None:
    bom = evaluate_validity_v2(ValidityInput(raw_bytes=b"\xef\xbb\xbfcomplete utf-8 payload"))
    undecodable = evaluate_validity_v2(ValidityInput(raw_bytes=b"useful prefix \x80 useful suffix"))
    binary = evaluate_validity_v2(ValidityInput(raw_bytes=b"\x00\x01\x02\x03" * 20))
    png = evaluate_validity_v2(ValidityInput(raw_bytes=b"\x89PNG\r\n\x1a\n" + b"payload"))

    assert bom.status.value == "valid_after_reversible_repair"
    assert "validity_utf8_bom_repaired" in bom.reason_codes
    assert bom.original_bytes_sha256 is not None
    assert undecodable.status.value == "quarantine"
    assert "validity_declared_decoding_failed" in undecodable.reason_codes
    assert undecodable.original_bytes == b"useful prefix \x80 useful suffix"
    assert binary.status.value == "invalid"
    assert "validity_binary_payload" in binary.reason_codes
    assert png.status.value == "invalid"

    ambiguous_case = next(case for case in load_cases() if case["id"] == "ambiguous_text_fields")
    ambiguous = evaluate_validity_v2(_input(ambiguous_case))
    assert tuple(field.name for field in ambiguous.original_text_fields) == ("text", "content")


def test_repairs_are_idempotent_and_rechunk_never_selects_directly() -> None:
    repaired = evaluate_validity_v2(ValidityInput.from_text("The bird\u0092s wing remains visible."))
    replay = evaluate_validity_v2(ValidityInput.from_text(repaired.recovered_text))
    rechunk = evaluate_validity_v2(
        ValidityInput.from_text(
            "```text\nA chunk ends early.",
            source_record_text="```text\nA chunk ends early.\n```",
        )
    )

    assert repaired.transformation_trace
    assert replay.status.value == "valid"
    assert replay.recovered_sha256 == repaired.recovered_sha256
    assert rechunk.requires_rechunk is True
    assert rechunk.training_eligible is False


def test_fixture_audit_reports_zero_observed_clean_false_positives() -> None:
    cases = tuple(
        ValidityAuditCase(
            case_id=str(case["id"]),
            role=str(case["role"]),
            input_unit=_input(case),
            expected_status=str(case["expected_status"]),
            expected_action=str(case["expected_action"]),
            expected_reason=str(case["expected_reason"]) if "expected_reason" in case else None,
        )
        for case in load_cases()
    )
    report = build_validity_audit(cases, confidence_level=0.95)

    assert report.clean_control_count >= 20
    assert report.clean_false_positive_count == 0
    assert report.clean_false_positive_upper_bound <= 0.15
    assert report.positive_false_negative_count == 0
    assert report.passed is True
    assert report.per_reason
    quarantine = next(result for result in report.case_results if result.case_id == "control_with_payload")
    assert quarantine.observed_action == "quarantine"
    assert "validity_forbidden_control_character" in quarantine.observed_reasons
    assert quarantine.original_field_hashes


def test_contract_keeps_v2_candidate_out_of_active_runtime() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["status"] == "block_3_fixture_validated_candidate"
    assert contract["runtime_activation"] is False
    assert contract["states"] == ["valid", "valid_after_reversible_repair", "quarantine", "invalid"]
    assert contract["provider_outputs_allowed"] is False
    assert contract["reject_boundary"] == "no_interpretable_text_control_only_bytes_or_registered_binary_magic"
    assert contract["binary_boundary"]["fractional_binary_threshold_used"] is False


if __name__ == "__main__":
    test_cross_domain_cases_follow_the_four_state_contract()
    test_raw_byte_boundaries_repair_quarantine_and_reject_without_guessing()
    test_repairs_are_idempotent_and_rechunk_never_selects_directly()
    test_fixture_audit_reports_zero_observed_clean_false_positives()
    test_contract_keeps_v2_candidate_out_of_active_runtime()
    print("[validity-v2] four-state cross-domain behavior and uncertainty audit: pass")
