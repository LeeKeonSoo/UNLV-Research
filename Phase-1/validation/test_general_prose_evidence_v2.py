#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_general_prose_evidence_v2 import (
    build_control,
    build_stress_variants,
    normalized_text_hash,
    paragraph_control_text,
    training_text_hashes,
)


def test_paragraph_control_is_deterministic_and_bounded() -> None:
    text = "First paragraph has useful context. " * 20 + "\n\n" + "Second paragraph. " * 40
    first = paragraph_control_text(text, minimum_chars=200, maximum_chars=700)
    second = paragraph_control_text(text, minimum_chars=200, maximum_chars=700)

    assert first == second
    assert 200 <= len(first) <= 700
    assert first.startswith("First paragraph")


def test_training_pair_hashes_cover_each_provider_text() -> None:
    hashes = training_text_hashes(
        (
            {"texts": ["First provider text.", "Second provider text."]},
            {"texts": ["Third provider text."]},
        )
    )

    assert hashes == {
        normalized_text_hash("First provider text."),
        normalized_text_hash("Second provider text."),
        normalized_text_hash("Third provider text."),
    }


def test_control_rejects_provider_training_overlap_and_non_general_route() -> None:
    prose = (
        "This historical account explains the evidence in a complete informational passage. "
        "It describes the relevant events and gives enough context for a reader to understand them."
    )
    overlap = {normalized_text_hash(prose)}
    assert build_control("source-a", "one", prose, overlap) is None
    assert build_control("source-a", "two", "def f(x):\n    return x + 1", set()) is None
    assert build_control("source-a", "bad-control", prose + "\u0081", set()) is None

    control = build_control("source-a", "three", prose, set())
    assert control is not None
    assert control["route_status"] == "routed"
    assert control["route_labels"] == ["general_prose"]


def test_stress_variants_preserve_linkage_and_name_the_expected_relation() -> None:
    base = {
        "chunk_uid": "source-a::one",
        "source_group": "source-a",
        "text": "A complete informational paragraph explains a historical event with supporting facts.",
    }
    variants = build_stress_variants(base)

    assert {row["variant"] for row in variants} == {
        "format_html",
        "format_markdown_quote",
        "semantic_destruction_token_permutation",
    }
    assert all(row["base_chunk_uid"] == base["chunk_uid"] for row in variants)
    relations = {row["variant"]: row["expected_relation"] for row in variants}
    assert relations["format_html"] == "retention_decision_invariant"
    assert relations["format_markdown_quote"] == "retention_decision_invariant"
    assert relations["semantic_destruction_token_permutation"] == "must_not_outscore_clean_pair"


if __name__ == "__main__":
    test_paragraph_control_is_deterministic_and_bounded()
    test_training_pair_hashes_cover_each_provider_text()
    test_control_rejects_provider_training_overlap_and_non_general_route()
    test_stress_variants_preserve_linkage_and_name_the_expected_relation()
    print("general prose evidence v2: ok")
