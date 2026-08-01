#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_general_prose_evidence_v2 import (
    GeneralScore,
    build_profile_report,
    select_strict_profile,
    source_balanced_thresholds,
)


def row(uid: str, source: str, facts: float, education: float) -> GeneralScore:
    return GeneralScore(uid, source, 100, facts, education)


def test_source_balanced_threshold_uses_each_source_not_pooled_average() -> None:
    rows = (
        row("a1", "a", 4.0, 5.0),
        row("a2", "a", 5.0, 6.0),
        row("b1", "b", -2.0, -3.0),
        row("b2", "b", -1.0, -2.0),
    )

    assert source_balanced_thresholds(rows, 0.0) == (-2.0, -3.0)


def test_leave_one_source_out_exposes_provider_source_bias() -> None:
    rows = tuple(row(f"a{i}", "a", 4.0, 5.0) for i in range(50)) + tuple(
        row(f"b{i}", "b", -2.0, -3.0) for i in range(50)
    )
    report = build_profile_report(rows, 0.0, 0.95)

    assert report["pooled_failures"] == 0
    assert report["leave_one_source_out"]["b"]["failures"] == 50
    assert select_strict_profile((report,), 0.05) is None


def test_strict_profile_requires_pooled_source_loso_and_length_gates() -> None:
    passing = {
        "profile_id": "q0",
        "quantile": 0.0,
        "pooled_wilson_upper_bound": 0.01,
        "max_source_wilson_upper_bound": 0.02,
        "max_leave_one_source_out_wilson_upper_bound": 0.03,
        "max_length_quartile_wilson_upper_bound": 0.02,
    }
    failing = {**passing, "profile_id": "q0.01", "quantile": 0.01, "max_length_quartile_wilson_upper_bound": 0.06}

    assert select_strict_profile((passing, failing), 0.05) == "q0"


if __name__ == "__main__":
    test_source_balanced_threshold_uses_each_source_not_pooled_average()
    test_leave_one_source_out_exposes_provider_source_bias()
    test_strict_profile_requires_pooled_source_loso_and_length_gates()
    print("general prose evidence audit v2: ok")
