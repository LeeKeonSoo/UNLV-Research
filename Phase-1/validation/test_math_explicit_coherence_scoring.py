#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_math_explicit_coherence import (
    BaseStructuralScore,
    GuardAuditContext,
    GuardGate,
    TextRow,
    build_guard_report,
    score_rows,
)


def test_explicit_coherence_scorer_preserves_payload_and_reason_codes() -> None:
    rows = (
        TextRow("good", "clean", "A complete theorem.", 4),
        TextRow("bad", "clean", "\\begin{proof}\nAn incomplete theorem.", 7),
    )
    base = {
        "good": BaseStructuralScore("good", 0.8, {"substantive_payload": "hash"}),
        "bad": BaseStructuralScore("bad", 0.7, {"substantive_payload": "hash"}),
    }

    scores = score_rows(rows, base)

    assert scores[0].substantive_payload == 0.8
    assert scores[0].coherence_completeness == 1.0
    assert scores[1].substantive_payload == 0.7
    assert scores[1].coherence_completeness == 0.0
    assert scores[1].reason_codes == ("coherence_unmatched_latex_environment",)


def test_clean_report_requires_source_and_fixture_gates() -> None:
    rows = tuple(TextRow(f"row-{index}", "clean", "A complete theorem.", 4) for index in range(100))
    base = {
        row.record_id: BaseStructuralScore(row.record_id, 0.8, {"substantive_payload": "hash"}) for row in rows
    }

    report = build_guard_report(
        score_rows(rows, base), GuardAuditContext("clean_control", GuardGate(0.95, 0.8, 0.05), "v2")
    )

    assert report["source_false_reject_gate_passed"] is True
    assert report["fixture_gate_passed"] is True
    assert report["status"] == "development_gates_passed_pending_fresh_controls"


def test_fresh_control_report_has_confirmatory_status() -> None:
    rows = tuple(TextRow(f"fresh-{index}", "fresh", "A complete theorem.", 4) for index in range(100))
    base = {
        row.record_id: BaseStructuralScore(row.record_id, 0.8, {"substantive_payload": "hash"}) for row in rows
    }

    report = build_guard_report(
        score_rows(rows, base), GuardAuditContext("fresh_control", GuardGate(0.95, 0.8, 0.05), "v2")
    )

    assert report["status"] == "fresh_control_gates_passed"


if __name__ == "__main__":
    test_explicit_coherence_scorer_preserves_payload_and_reason_codes()
    test_clean_report_requires_source_and_fixture_gates()
    test_fresh_control_report_has_confirmatory_status()
    print("[math-explicit-coherence] scoring and audit gates: pass")
