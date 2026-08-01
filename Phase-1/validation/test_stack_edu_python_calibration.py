#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_stack_edu_python import (
    StackEduScore,
    build_source_balanced_profiles,
    select_strict_profile,
)


def row(uid: str, source: str, score: float, tokens: int = 10) -> StackEduScore:
    return StackEduScore(uid, uid.encode().hex().ljust(64, "0")[:64], source, tokens, score)


def test_threshold_is_minimum_source_quantile_not_pooled_quantile() -> None:
    clean = (
        row("a", "small", 0.5),
        row("b", "small", 1.0),
        row("c", "large", 2.0),
        row("d", "large", 3.0),
    )

    profiles = build_source_balanced_profiles(clean, (0.0, 0.5))

    assert profiles[0].routes[0].route_specific_evidence == 0.5
    assert profiles[1].routes[0].route_specific_evidence == 0.5


def test_leave_one_source_out_failure_blocks_profile() -> None:
    strict_profiles = (
        {
            "profile_id": "clean_q0",
            "pooled_wilson_upper_bound": 0.01,
            "max_source_wilson_upper_bound": 0.04,
            "max_leave_one_source_out_wilson_upper_bound": 0.08,
            "excluded_candidate_tokens": 100,
        },
    )

    assert select_strict_profile(strict_profiles, 0.05) is None


if __name__ == "__main__":
    test_threshold_is_minimum_source_quantile_not_pooled_quantile()
    test_leave_one_source_out_failure_blocks_profile()
    print("[stack-edu-python-calibration] source-balanced strict gate: pass")
