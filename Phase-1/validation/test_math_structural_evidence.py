#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explicit_structural_coherence import explicit_coherence_evidence
from math_structural_evidence import coherence_corruptions, extract_structural_features, payload_corruptions


TEXT = """Quadratic equations
For a nonzero coefficient a, solve ax^2 + bx + c = 0 by completing the square.
The solutions satisfy x = (-b +/- sqrt(b^2 - 4ac)) / (2a)."""


def test_payload_corruption_preserves_scale_but_increases_repetition() -> None:
    corrupted = payload_corruptions(TEXT, "stable-id")

    original_features = extract_structural_features(TEXT)
    repeated_features = extract_structural_features(corrupted[0].text)

    assert len(corrupted) == 2
    assert repeated_features.lexical_tokens >= original_features.lexical_tokens * 0.9
    assert repeated_features.repeated_line_fraction > original_features.repeated_line_fraction


def test_coherence_corruptions_are_deterministic_and_damage_boundaries() -> None:
    first = coherence_corruptions(TEXT, "stable-id")
    second = coherence_corruptions(TEXT, "stable-id")

    assert first == second
    assert len(first) == 3
    assert any(not extract_structural_features(item.text).terminal_boundary for item in first)


def test_unbalanced_delimiter_corruption_lowers_balance() -> None:
    text = "A proof uses (x + y), [x - y], and {x, y}."
    corrupted = coherence_corruptions(text, "delimiter-id")

    original = extract_structural_features(text)
    damaged = extract_structural_features(corrupted[-1].text)

    assert original.delimiter_balance == 1.0
    assert damaged.delimiter_balance < original.delimiter_balance


def test_v2_features_detect_markup_and_document_boundary_damage() -> None:
    complete = extract_structural_features("\\begin{proof}\nThe claim follows.\n\\end{proof}")
    damaged = extract_structural_features("\\begin{proof}\nthe claim follows")

    assert complete.markup_pair_balance > damaged.markup_pair_balance
    assert complete.boundary_completeness > damaged.boundary_completeness


def test_v2_vector_adds_repetition_and_boundary_evidence_without_changing_v1() -> None:
    features = extract_structural_features("Repeat this statement.\nRepeat this statement.\nRepeat this statement.")

    assert len(features.vector()) == 11
    assert len(features.vector("v2")) == 15
    assert features.repeated_ngram_fraction > 0.0


def test_explicit_coherence_guard_rejects_only_registered_observable_damage() -> None:
    fixtures = {
        "coherence_unicode_replacement_burst": "A valid statement.\ufffd\ufffd\ufffd",
        "coherence_forbidden_control_character": "A valid\x00 statement.",
        "coherence_unmatched_latex_environment": "\\begin{proof}\nThe claim follows.",
        "coherence_unmatched_explicit_xml_tag": "<theorem>The claim follows.",
        "coherence_dangling_markdown_fence": "```math\nx^2 + y^2 = 1",
        "coherence_repeated_delimiter_damage": "A derivation remains visible.\n((( [[ {{{",
    }

    for expected_reason, text in fixtures.items():
        evidence = explicit_coherence_evidence(text)
        assert evidence.outcome == "explicit_defect"
        assert expected_reason in evidence.reason_codes


def test_explicit_coherence_guard_preserves_balanced_and_non_identifiable_text() -> None:
    retained = (
        "\\begin{proof}\nThe claim follows.\n\\end{proof}",
        "<theorem>The claim follows.</theorem>",
        "```math\nx^2 + y^2 = 1\n```",
        "a lowercase fragment without terminal punctuation",
        "Second half. First half.",
        "% <*SimilarityMotivation>\n% </SimilarityMotivation>\n\\section{Similarity}\nSubstantive mathematical prose.",
        "For every i<j and vector v<R, the relation remains meaningful.",
        "\\end{input}\nA closing construct alone does not prove truncation.",
        "Internal notation ((( can remain meaningful when later prose follows.",
    )

    assert all(explicit_coherence_evidence(text).outcome == "guard_passed" for text in retained)


if __name__ == "__main__":
    test_payload_corruption_preserves_scale_but_increases_repetition()
    test_coherence_corruptions_are_deterministic_and_damage_boundaries()
    test_unbalanced_delimiter_corruption_lowers_balance()
    test_v2_features_detect_markup_and_document_boundary_damage()
    test_v2_vector_adds_repetition_and_boundary_evidence_without_changing_v1()
    test_explicit_coherence_guard_rejects_only_registered_observable_damage()
    test_explicit_coherence_guard_preserves_balanced_and_non_identifiable_text()
    print("[math-structural-evidence] metamorphic features: pass")
