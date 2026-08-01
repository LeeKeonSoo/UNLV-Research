#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from composition_audit import annotate_record, annotate_records, build_composition_audit


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FIXTURES = ROOT / "validation" / "fixtures" / "coverage_taxonomy_v1_cases.json"
TAXONOMY = ROOT / "configs" / "coverage_taxonomy_v1.json"


def load_fixtures() -> dict[str, JsonValue]:
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def test_multilabel_axes_and_unknown_are_explicit() -> None:
    fixtures = load_fixtures()
    cases = fixtures["cases"]
    assert isinstance(cases, list)

    for case in cases:
        assert isinstance(case, dict)
        annotation = annotate_record({"text": case["text"]})["coverage_v1"]
        expected = case.get("expected", {})
        assert isinstance(expected, dict)
        for axis, labels in expected.items():
            actual_labels = set(annotation[axis]["labels"])
            assert set(labels) <= actual_labels
            assert annotation[axis]["status"] == "classified"
        for axis in case.get("expected_unknown_axes", []):
            assert annotation[axis]["labels"] == ["unknown"]
            assert annotation[axis]["status"] == "unknown"


def test_source_metadata_cannot_change_coverage_tags() -> None:
    text = "def solve(value):\n    return value * value\nThe equation describes the result."
    first = annotate_record({"text": text, "source_name": "source-a"})["coverage_v1"]
    second = annotate_record({"text": text, "source_name": "source-b", "declared_domain": "law"})["coverage_v1"]

    assert first == second


def test_emitted_labels_are_registered_and_have_no_selection_authority() -> None:
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    fixtures = load_fixtures()
    cases = fixtures["cases"]
    assert isinstance(cases, list)

    assert taxonomy["selection_authority"] is False
    assert taxonomy["quota_authority"] is False
    assert taxonomy["cross_stratum_importance_authority"] is False
    for case in cases:
        assert isinstance(case, dict)
        annotation = annotate_record({"text": case["text"]})["coverage_v1"]
        for axis, registered_labels in taxonomy["axes"].items():
            assert set(annotation[axis]["labels"]) <= set(registered_labels)


def test_coverage_distribution_is_multilabel_and_audit_only() -> None:
    raw = annotate_records(
        [
            {"text": "def add(a, b):\n    return a + b\nimport math", "token_proxy": 8},
            {"text": "@@@ ### 12345 === ???", "token_proxy": 5},
        ]
    )
    curated = annotate_records([{"text": "def add(a, b):\n    return a + b\nimport math", "token_proxy": 8}])
    audit = build_composition_audit({"raw_input": raw, "stage_c_curated": curated})
    coverage = audit["coverage_v1"]

    assert coverage["authority"] == "audit_only"
    assert coverage["consumed_by_selection"] is False
    assert coverage["classification"] == "multi_label_with_unknown"
    assert coverage["stages"]["raw_input"]["semantic_domain"]["records"]["code"] == 1
    assert coverage["stages"]["raw_input"]["semantic_domain"]["records"]["unknown"] == 1
    assert coverage["stages"]["raw_input"]["semantic_domain"]["unknown_record_rate"] == 0.5
    assert "semantic_domain" in coverage["jensen_shannon_divergence_from_raw"]["stage_c_curated"]


if __name__ == "__main__":
    test_multilabel_axes_and_unknown_are_explicit()
    test_source_metadata_cannot_change_coverage_tags()
    test_emitted_labels_are_registered_and_have_no_selection_authority()
    test_coverage_distribution_is_multilabel_and_audit_only()
    print("[coverage-taxonomy-v1] multi-label audit taxonomy: pass")
