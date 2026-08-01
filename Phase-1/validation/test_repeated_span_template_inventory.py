#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repeated_span_template_inventory import build_inventory


def test_repeated_span_inventory_detects_only_exact_long_cross_record_boilerplate() -> None:
    shared = "This generated reference section describes the shared transport protocol and applies unchanged to every client package."
    report = build_inventory(
        [
            {"record_id": "one", "text": f"{shared}\n\nPayload one explains retry behavior for a specific API."},
            {"record_id": "two", "text": f"{shared}\n\nPayload two explains authentication behavior for a different API."},
            {"record_id": "three", "text": "This generated reference section describes a different protocol with distinct client behavior and details."},
        ],
        minimum_lexical_tokens=12,
    )

    assert report["status"] == "diagnostic_only_not_a_selection_policy"
    assert report["repeated_span_family_count"] == 1
    family = report["families"][0]
    assert family["record_ids"] == ["one", "two"]
    assert family["member_count"] == 2
    assert family["span_token_proxy"] >= 12
    assert report["selector_consumes_this_inventory"] is False


def test_repeated_span_inventory_counts_a_repeated_span_once_per_record() -> None:
    shared = "This repeated section is long enough to qualify as a diagnostic template family across independent records."
    report = build_inventory(
        [
            {"record_id": "one", "text": f"{shared}\n\n{shared}"},
            {"record_id": "two", "text": shared},
        ],
        minimum_lexical_tokens=12,
    )

    assert report["families"][0]["record_ids"] == ["one", "two"]
    assert report["families"][0]["member_count"] == 2


def test_repeated_span_inventory_uses_stage_b_chunk_uid_when_record_id_is_absent() -> None:
    shared = "This repeated section is long enough to qualify as a diagnostic template family across independent records."
    report = build_inventory(
        [
            {"chunk_uid": "one::0000", "text": f"{shared}\n\nFirst independent payload remains useful."},
            {"chunk_uid": "two::0000", "text": f"{shared}\n\nSecond independent payload remains useful."},
        ],
        minimum_lexical_tokens=12,
    )

    assert report["repeated_span_family_count"] == 1
    assert report["families"][0]["record_ids"] == ["one::0000", "two::0000"]


def test_repeated_span_inventory_rejects_short_common_phrases_and_payload_variants() -> None:
    report = build_inventory(
        [
            {"record_id": "short-one", "text": "See documentation for details.\n\nFirst payload."},
            {"record_id": "short-two", "text": "See documentation for details.\n\nSecond payload."},
            {"record_id": "variant-one", "text": "The invoice identifier is 100 and the customer account is alpha."},
            {"record_id": "variant-two", "text": "The invoice identifier is 200 and the customer account is beta."},
        ],
        minimum_lexical_tokens=12,
    )

    assert report["repeated_span_family_count"] == 0


if __name__ == "__main__":
    test_repeated_span_inventory_detects_only_exact_long_cross_record_boilerplate()
    test_repeated_span_inventory_counts_a_repeated_span_once_per_record()
    test_repeated_span_inventory_uses_stage_b_chunk_uid_when_record_id_is_absent()
    test_repeated_span_inventory_rejects_short_common_phrases_and_payload_variants()
    print("[repeated-span-template-inventory] exact long-span diagnostic boundary: pass")
