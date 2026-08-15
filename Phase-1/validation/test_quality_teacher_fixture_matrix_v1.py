#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_fixtures import (
    FixtureClass,
    build_behavior_fixture_matrix,
    build_protected_fixture_set,
    build_ranker_enrichment_fixture_set,
)
from quality_teacher_fixture_dataset import materialize


def test_behavior_matrix_fills_every_frozen_cell() -> None:
    fixtures = build_behavior_fixture_matrix(samples_per_cell=8)

    assert len(fixtures) == 512
    assert len({fixture.fixture_id for fixture in fixtures}) == 512
    cells = Counter(
        (fixture.policy_id, fixture.route, fixture.fixture_class) for fixture in fixtures
    )
    assert len(cells) == 4 * 4 * 4
    assert set(cells.values()) == {8}
    assert {fixture.fixture_class for fixture in fixtures} == set(FixtureClass)
    assert all(fixture.label_provenance == "deterministic_construction" for fixture in fixtures)
    assert all("benchmark" not in fixture.unit.declared_context.lower() for fixture in fixtures)


def test_protected_set_has_eight_hundred_verifiable_quality_passes() -> None:
    fixtures = build_protected_fixture_set(samples_per_route=200)

    assert len(fixtures) == 800
    assert len({fixture.fixture_id for fixture in fixtures}) == 800
    assert Counter(fixture.route for fixture in fixtures) == {
        "code_artifact": 200,
        "mathematical_content": 200,
        "general_prose": 200,
        "table_structured_data": 200,
    }
    assert all(fixture.expected_quality_gate == "pass" for fixture in fixtures)
    assert all(len(fixture.verifier_evidence) >= 1 for fixture in fixtures)


def test_ranker_enrichment_has_unique_text_and_balanced_target_policy_classes() -> None:
    fixtures = build_ranker_enrichment_fixture_set(samples_per_cell=12)

    assert len(fixtures) == 576
    assert len({fixture.fixture_id for fixture in fixtures}) == 576
    assert len({fixture.unit.text for fixture in fixtures}) == 576
    cells = Counter((fixture.policy_id, fixture.fixture_class) for fixture in fixtures)
    assert len(cells) == 4 * 3
    assert set(cells.values()) == {48}
    assert {fixture.fixture_class for fixture in fixtures} == {
        FixtureClass.PASS,
        FixtureClass.FAIL,
        FixtureClass.ABSTAIN,
    }


def test_materialized_ranker_enrichment_preserves_target_policy_and_verifier() -> None:
    with TemporaryDirectory() as temporary_directory:
        audit = materialize(Path(temporary_directory))
        enrichment_path = Path(str(audit["ranker_enrichment_path"]))
        rows = [
            json.loads(line)
            for line in enrichment_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    assert len(rows) == 576
    assert len({row["text"] for row in rows}) == 576
    cells = Counter((row["fixture_policy_id"], row["fixture_class"]) for row in rows)
    assert len(cells) == 12
    assert set(cells.values()) == {48}
    q1_decisive = [
        row
        for row in rows
        if row["fixture_policy_id"] == "q1_correctness_evidence"
        and row["fixture_class"] in {"pass", "fail"}
    ]
    assert len(q1_decisive) == 96
    assert all(row["quality_declared_verifier"] is not None for row in q1_decisive)


if __name__ == "__main__":
    test_behavior_matrix_fills_every_frozen_cell()
    test_protected_set_has_eight_hundred_verifiable_quality_passes()
    test_ranker_enrichment_has_unique_text_and_balanced_target_policy_classes()
    test_materialized_ranker_enrichment_preserves_target_policy_and_verifier()
    print("[quality-teacher-fixtures-v1] 512 behavior and 800 protected fixtures: pass")
