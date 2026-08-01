#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_math_structural_heads import TextRow, feature_vector, split_source_roles


def row(uid: str, source: str) -> TextRow:
    return TextRow(uid, source, f"Substantive mathematical explanation for {uid}. " * 10, 100)


def test_source_roles_are_disjoint_and_complete() -> None:
    rows = (row("a", "train"), row("b", "cal-one"), row("c", "cal-two"))

    training, calibration = split_source_roles(rows, frozenset({"train"}), frozenset({"cal-one", "cal-two"}))

    assert {item.record_id for item in training} == {"a"}
    assert {item.record_id for item in calibration} == {"b", "c"}


def test_undeclared_source_is_rejected() -> None:
    rows = (row("a", "train"), row("x", "undeclared"))

    try:
        split_source_roles(rows, frozenset({"train"}), frozenset({"cal"}))
    except ValueError as error:
        assert "undeclared" in str(error)
    else:
        raise AssertionError("Expected undeclared clean-control source to fail")


def test_feature_schema_version_selects_frozen_vector_shape() -> None:
    text = "A complete mathematical statement."

    assert len(feature_vector(text, "v1")) == 11
    assert len(feature_vector(text, "v2")) == 15


if __name__ == "__main__":
    test_source_roles_are_disjoint_and_complete()
    test_undeclared_source_is_rejected()
    test_feature_schema_version_selects_frozen_vector_shape()
    print("[math-structural-heads] source-role split: pass")
