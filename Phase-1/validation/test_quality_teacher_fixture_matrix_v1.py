#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_fixtures import (
    FixtureClass,
    build_behavior_fixture_matrix,
    build_protected_fixture_set,
)


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


if __name__ == "__main__":
    test_behavior_matrix_fills_every_frozen_cell()
    test_protected_set_has_eight_hundred_verifiable_quality_passes()
    print("[quality-teacher-fixtures-v1] 512 behavior and 800 protected fixtures: pass")
