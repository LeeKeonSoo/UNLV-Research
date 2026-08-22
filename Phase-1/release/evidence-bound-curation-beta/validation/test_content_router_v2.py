#!/usr/bin/env python3
from __future__ import annotations

import json
import inspect
import sys
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_router import route_content


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FIXTURES = ROOT / "validation" / "fixtures" / "content_router_v2_cases.json"


def load_cases() -> list[dict[str, JsonValue]]:
    payload = json.loads(FIXTURES.read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert isinstance(cases, list)
    return cases


def test_registered_cases_have_expected_routes_and_axes() -> None:
    for case in load_cases():
        text = case["text"]
        assert isinstance(text, str)
        result = route_content(text)

        assert set(case["expected_routes"]) <= set(result["route_labels"])
        assert result["route_status"] == case["expected_status"]
        assert result["route_confidence"] == case["expected_confidence"]
        expected_axes = case["expected_axes"]
        assert isinstance(expected_axes, dict)
        for axis, labels in expected_axes.items():
            assert set(labels) <= set(result[axis]["labels"])


def test_router_has_metadata_authority_only() -> None:
    result = route_content("def solve(value):\n    return value + 1\nimport math")

    assert result["authority"] == "shared_observable_metadata_only"
    assert result["may_select_or_remove"] is False
    assert result["may_assign_importance"] is False
    assert "quality_decision" not in result
    assert "retention_decision" not in result


def test_source_metadata_cannot_change_routing() -> None:
    assert tuple(inspect.signature(route_content).parameters) == ("text",)


def test_math_prose_and_code_comments_do_not_become_ambiguous_by_default() -> None:
    math = route_content(
        "The theorem follows from the matrix equation. Proof: use \\frac{1}{2} and \\sum_i x_i."
    )
    code = route_content(
        "# This function explains the conversion.\nimport math\ndef convert(value):\n    return math.floor(value)"
    )

    assert math["route_labels"] == ["mathematical_content"]
    assert math["route_status"] == "routed"
    assert code["route_labels"] == ["code_artifact"]
    assert code["route_status"] == "routed"


def test_multilingual_scripts_are_registered_instead_of_marked_unsupported() -> None:
    cases = {
        "greek": "Αυτό είναι ελληνικό κείμενο.",
        "hebrew": "זהו טקסט בעברית.",
        "thai": "นี่คือข้อความภาษาไทย",
        "bengali": "এটি একটি বাংলা পাঠ্য।",
        "tamil": "இது ஒரு தமிழ் உரை.",
        "telugu": "ఇది తెలుగు వచనం.",
    }

    for expected_script, sample in cases.items():
        result = route_content(sample)
        assert expected_script in result["language_script"]["labels"]
        assert result["language_script"]["status"] != "out_of_distribution"


if __name__ == "__main__":
    test_registered_cases_have_expected_routes_and_axes()
    test_router_has_metadata_authority_only()
    test_source_metadata_cannot_change_routing()
    test_math_prose_and_code_comments_do_not_become_ambiguous_by_default()
    test_multilingual_scripts_are_registered_instead_of_marked_unsupported()
    print("[content-router-v2] observable routing without selection authority: pass")
