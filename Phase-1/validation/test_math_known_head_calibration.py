#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_math_known_heads import MathKnownScore, build_source_balanced_profiles


def row(uid: str, source: str, route: float, usefulness: float) -> MathKnownScore:
    return MathKnownScore(uid, uid.encode().hex().ljust(64, "0")[:64], source, 10, route, usefulness)


def test_each_threshold_uses_the_least_restrictive_source_quantile() -> None:
    controls = (
        row("a", "low", 0.2, 1.0),
        row("b", "low", 0.4, 2.0),
        row("c", "high", 0.8, 3.0),
        row("d", "high", 1.0, 4.0),
    )

    profiles = build_source_balanced_profiles(controls, (0.5,))

    assert profiles[0].route_threshold == 0.2
    assert profiles[0].usefulness_threshold == 1.0


def test_profile_grid_is_frozen_cartesian_product() -> None:
    controls = (
        row("a", "one", 0.2, 1.0),
        row("b", "one", 0.4, 2.0),
        row("c", "two", 0.8, 3.0),
        row("d", "two", 1.0, 4.0),
    )

    profiles = build_source_balanced_profiles(controls, (0.0, 0.5))

    assert len(profiles) == 4
    assert {profile.profile_id for profile in profiles} == {
        "route_q0__usefulness_q0",
        "route_q0__usefulness_q0.5",
        "route_q0.5__usefulness_q0",
        "route_q0.5__usefulness_q0.5",
    }


if __name__ == "__main__":
    test_each_threshold_uses_the_least_restrictive_source_quantile()
    test_profile_grid_is_frozen_cartesian_product()
    print("[math-known-heads] source-balanced profile construction: pass")
