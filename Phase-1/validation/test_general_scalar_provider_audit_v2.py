#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_general_scalar_provider_v2 import (
    ScalarScore,
    build_scalar_profile_report,
    decide_scalar_provider_status,
    select_scalar_profile,
)


def row(uid: str, source: str, score: float) -> ScalarScore:
    return ScalarScore(uid, source, 100, score)


def test_leave_one_source_out_blocks_source_specific_scalar_boundary() -> None:
    rows = tuple(row(f"a{i}", "a", 0.9) for i in range(50)) + tuple(
        row(f"b{i}", "b", 0.1) for i in range(50)
    )

    report = build_scalar_profile_report(rows, 0.0, 0.95)

    assert report["pooled_failures"] == 0
    assert report["leave_one_source_out"]["b"]["failures"] == 50
    assert select_scalar_profile((report,), 0.05) is None


def test_component_pass_does_not_claim_complete_quality_bundle() -> None:
    rows = (
        row("a1", "a", 0.4),
        row("a2", "a", 0.5),
        row("b1", "b", 0.4),
        row("b2", "b", 0.5),
    )

    report = build_scalar_profile_report(rows, 0.0, 0.95)

    assert report["evidence_head"] == "route_specific_evidence"
    assert report["complete_quality_bundle"] is False


def test_status_reports_every_blocking_gate() -> None:
    stress = {
        "max_format_flip_wilson_upper_bound": 0.01,
        "semantic_destruction": {"wilson_upper_bound": 0.40},
    }

    decision = decide_scalar_provider_status(None, None, stress, 0.05)

    assert decision["status"] == "blocked_multiple_gates"
    assert decision["blocking_gates"] == ["source_transfer", "semantic_destruction"]


if __name__ == "__main__":
    test_leave_one_source_out_blocks_source_specific_scalar_boundary()
    test_component_pass_does_not_claim_complete_quality_bundle()
    test_status_reports_every_blocking_gate()
    print("general scalar provider audit v2: ok")
