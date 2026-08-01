#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_math_hard_candidate_profile import (
    HeadThresholds,
    FixtureGate,
    FixtureObservation,
    failed_head_names,
    summarize_ablation_arms,
    summarize_fixture_family,
)
from scripts.calibrate_math_complete_bundle import CompleteMathScore


THRESHOLDS = HeadThresholds(0.2, 0.3, 0.4, 0.5)


def row(uid: str, tokens: int, scores: tuple[float, float, float, float]) -> CompleteMathScore:
    return CompleteMathScore(uid, uid.encode().hex().ljust(64, "0")[:64], "fixture", tokens, *scores)


def test_failed_heads_are_reported_independently() -> None:
    observed = failed_head_names(row("a", 10, (0.1, 0.35, 0.2, 0.7)), THRESHOLDS)

    assert observed == ("route_confidence", "coherence_completeness")


def test_ablation_reports_incremental_record_and_token_exclusion() -> None:
    rows = (
        row("known", 10, (0.1, 0.9, 0.9, 0.9)),
        row("payload", 20, (0.9, 0.1, 0.9, 0.9)),
        row("coherence", 30, (0.9, 0.9, 0.1, 0.9)),
        row("keep", 40, (0.9, 0.9, 0.9, 0.9)),
    )

    reports = summarize_ablation_arms(rows, THRESHOLDS)

    assert reports[0]["excluded_records"] == 1
    assert reports[0]["excluded_tokens"] == 10
    assert reports[-1]["excluded_records"] == 3
    assert reports[-1]["excluded_tokens"] == 60


def test_fixture_gate_uses_one_sided_wilson_sensitivity_lower_bound() -> None:
    gate = FixtureGate(0.95, 0.9)
    passing = summarize_fixture_family(FixtureObservation("damage", 99, 100), gate)
    failing = summarize_fixture_family(FixtureObservation("damage", 80, 100), gate)

    assert passing["gate_passed"] is True
    assert failing["gate_passed"] is False
    assert float(failing["wilson_sensitivity_lower_bound"]) < 0.9


if __name__ == "__main__":
    test_failed_heads_are_reported_independently()
    test_ablation_reports_incremental_record_and_token_exclusion()
    test_fixture_gate_uses_one_sided_wilson_sensitivity_lower_bound()
    print("[math-hard-candidate-audit] reason, ablation, and fixture gates: pass")
