#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from math_positive_evidence import build_math_candidate_evidence, has_explicit_math_notation


def test_two_providers_remain_independent_and_cannot_emit_keep() -> None:
    seen: list[tuple[str, str]] = []

    def relevance(text: str) -> float:
        seen.append(("relevance", text))
        return 0.83

    def usefulness(text: str) -> float:
        seen.append(("usefulness", text))
        return 3.4

    evidence = build_math_candidate_evidence("A proof of the identity.", relevance, usefulness)

    assert evidence.route_confidence == 0.83
    assert evidence.mathscore_probability == 0.83
    assert evidence.explicit_math_notation is False
    assert evidence.route_specific_evidence == 3.4
    assert evidence.substantive_payload is None
    assert evidence.coherence_completeness is None
    assert evidence.can_emit_keep is False
    assert seen == [
        ("relevance", "A proof of the identity."),
        ("usefulness", "A proof of the identity."),
    ]


def test_closed_math_notation_can_establish_route_without_copying_usefulness() -> None:
    assert has_explicit_math_notation("We obtain $$x^2 + y^2 = 1$$.") is True
    assert has_explicit_math_notation("The ticket costs $5 today.") is False

    evidence = build_math_candidate_evidence(
        "We obtain $$x^2 + y^2 = 1$$.",
        lambda _: 0.2,
        lambda _: 2.5,
    )

    assert evidence.route_confidence == 1.0
    assert evidence.mathscore_probability == 0.2
    assert evidence.route_specific_evidence == 2.5


if __name__ == "__main__":
    test_two_providers_remain_independent_and_cannot_emit_keep()
    test_closed_math_notation_can_establish_route_without_copying_usefulness()
    print("[math-positive-evidence] independent incomplete bundle: pass")
