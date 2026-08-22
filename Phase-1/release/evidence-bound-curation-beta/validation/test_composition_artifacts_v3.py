#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from composition_artifacts import (
    CompositionArtifactError,
    CompositionRecord,
    build_composition_artifacts,
    write_composition_artifacts,
)


def _records() -> tuple[CompositionRecord, ...]:
    return (
        CompositionRecord("code-ko", "# 설명\nimport math\ndef solve(x):\n    return x + 1", 40),
        CompositionRecord("math-ar", "نظرية ومعادلة theorem matrix equation proof \\frac{1}{2}", 30),
        CompositionRecord("prose-en", "This document explains a reproducible scientific method. Therefore the conclusion follows.", 30),
    )


def test_primary_routes_sum_to_one_and_multilabel_incidence_is_audit_only() -> None:
    audit = build_composition_artifacts(_records(), _records()[:2])

    raw_primary = [item for item in audit.shares if item.stage == "eligible" and item.axis == "primary_route"]
    raw_multilabel = [item for item in audit.shares if item.stage == "eligible" and item.axis == "route_incidence"]

    assert sum(item.token_share for item in raw_primary) == 1.0
    assert sum(item.token_share for item in raw_multilabel) >= 1.0
    assert audit.authority == "audit_only"
    assert audit.consumed_by_selection is False
    assert audit.target_distribution_enforced is False


def test_json_and_csv_artifacts_are_written_with_eligible_curated_deltas() -> None:
    audit = build_composition_artifacts(_records(), _records()[:2])

    with tempfile.TemporaryDirectory() as directory:
        paths = write_composition_artifacts(audit, Path(directory))
        payload = json.loads(paths.audit_json.read_text(encoding="utf-8"))
        with paths.route_csv.open(encoding="utf-8", newline="") as handle:
            route_rows = list(csv.DictReader(handle))
        with paths.delta_csv.open(encoding="utf-8", newline="") as handle:
            delta_rows = list(csv.DictReader(handle))

        assert set(paths.all()) == {
            Path(directory) / "composition_audit.json",
            Path(directory) / "composition_by_route.csv",
            Path(directory) / "composition_by_language.csv",
            Path(directory) / "eligible_curated_composition_delta.csv",
        }
        assert payload["authority"] == "audit_only"
        assert payload["comparison_unit"] == "chunk"
        assert payload["baseline_stage"] == "stage_b_pass"
        assert {row["stage"] for row in route_rows} == {"eligible", "curated"}
        assert any(row["axis"] == "primary_route" for row in delta_rows)


def test_mixed_content_is_not_arbitrarily_reported_as_one_specialized_route() -> None:
    mixed = CompositionRecord(
        "mixed",
        "Ava: Inspect this function.\nBen: I will run it.\nAva: Send the result.\n"
        "import math\ndef solve(value):\n    return value + 1",
        20,
    )

    audit = build_composition_artifacts((mixed,), (mixed,))
    primary = [
        item
        for item in audit.shares
        if item.stage == "eligible" and item.axis == "primary_route"
    ]

    assert [(item.label, item.token_share) for item in primary] == [("mixed", 1.0)]


def test_curated_ids_must_exist_in_the_eligible_chunk_baseline() -> None:
    eligible = (CompositionRecord("eligible", "def keep():\n    return 1", 5),)
    curated = (CompositionRecord("unrelated", "def other():\n    return 2", 5),)

    try:
        build_composition_artifacts(eligible, curated)
    except CompositionArtifactError as error:
        assert str(error) == "Curated composition IDs must be a subset of eligible chunk IDs"
    else:
        raise AssertionError("Cross-unit composition comparisons must be rejected")


if __name__ == "__main__":
    test_primary_routes_sum_to_one_and_multilabel_incidence_is_audit_only()
    test_json_and_csv_artifacts_are_written_with_eligible_curated_deltas()
    test_mixed_content_is_not_arbitrarily_reported_as_one_specialized_route()
    test_curated_ids_must_exist_in_the_eligible_chunk_baseline()
    print("[composition-artifacts-v3] explanatory outputs only: pass")
