#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_math_complete_bundle import (
    CompleteMathScore,
    build_source_balanced_profiles,
    leave_one_source_out_diagnostics,
    select_declared_calibration_sources,
)


def row(uid: str, source: str, offset: float) -> CompleteMathScore:
    return CompleteMathScore(uid, uid.encode().hex().ljust(64, "0")[:64], source, 10, offset, offset + 1, offset + 2, offset + 3)


def test_four_head_profile_uses_each_sources_lower_tail() -> None:
    rows = (row("a", "low", 0.1), row("b", "low", 0.2), row("c", "high", 0.8), row("d", "high", 0.9))

    profiles = build_source_balanced_profiles(rows, (0.5,))

    assert len(profiles) == 1
    assert profiles[0].thresholds == (0.1, 1.1, 2.1, 3.1)


def test_profile_grid_covers_every_frozen_quantile_combination() -> None:
    rows = (row("a", "one", 0.1), row("b", "two", 0.2))

    profiles = build_source_balanced_profiles(rows, (0.0, 0.5))

    assert len(profiles) == 16


def test_loso_diagnostics_identify_source_and_head_scale_shift() -> None:
    rows = (
        row("a", "high", 0.5),
        row("b", "high", 0.6),
        row("c", "low", 0.1),
        row("d", "low", 0.2),
    )
    profile = build_source_balanced_profiles(rows, (0.0,))[0]

    diagnostics = leave_one_source_out_diagnostics(rows, profile, 0.95)

    assert diagnostics["high"]["failures"] == 0
    assert diagnostics["low"]["failures"] == 2
    assert diagnostics["low"]["per_head_failures"]["route_confidence"] == 2


def test_calibration_source_selection_allows_only_declared_non_calibration_inputs() -> None:
    rows = (row("a", "calibration", 0.1), row("b", "training", 0.2))

    selected = select_declared_calibration_sources(rows, frozenset({"calibration"}), frozenset({"training"}))

    assert tuple(item.source_group for item in selected) == ("calibration",)


if __name__ == "__main__":
    test_four_head_profile_uses_each_sources_lower_tail()
    test_profile_grid_covers_every_frozen_quantile_combination()
    test_loso_diagnostics_identify_source_and_head_scale_shift()
    test_calibration_source_selection_allows_only_declared_non_calibration_inputs()
    print("[math-complete-bundle] four-head profile construction: pass")
