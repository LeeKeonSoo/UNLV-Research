#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_coverage_empirical_audit import EmpiricalCoverageTag, build_empirical_audit


def test_empirical_gate_requires_determinism_extinction_recall_and_zero_false_veto() -> None:
    uids = ("code-a", "code-b", "math-a", "math-b", "tail")
    primary = np.asarray(
        [[1.0, 0.0, 0.0], [0.99, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.99, 0.01], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    audit = np.asarray(
        [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0], [0.0, 1.0, 0.0], [0.0, 0.98, 0.02], [0.7, 0.0, 0.7]],
        dtype=np.float32,
    )

    report = build_empirical_audit(
        uids=uids,
        primary_vectors=primary,
        audit_vectors=audit,
        neighbor_count=1,
        block_size=2,
        device="cpu",
        corpus_sha256="3" * 64,
        primary_identity_sha256="1" * 64,
        audit_identity_sha256="2" * 64,
        tags=(
            EmpiricalCoverageTag("code-a", ("code_artifact",), ("latin",)),
            EmpiricalCoverageTag("code-b", ("code_artifact",), ("latin",)),
            EmpiricalCoverageTag("math-a", ("mathematical_content",), ("latin",)),
            EmpiricalCoverageTag("math-b", ("mathematical_content",), ("arabic",)),
            EmpiricalCoverageTag("tail", ("unknown",), ("unknown",)),
        ),
    )

    assert report.deterministic_replay is True
    assert report.contract_extinction_detection_recall == 1.0
    assert report.contract_representative_preserving_false_veto_rate == 0.0
    assert report.descriptive_bias_slices_complete is True
    assert report.stable_strata >= 2
    assert {cell.label for cell in report.route_agreement} == {
        "code_artifact",
        "mathematical_content",
        "unknown",
    }
    assert report.implementation_gate_passed is True
    assert report.scientific_promotion_gate_passed is False
    assert "protected_false_veto_evidence_missing" in report.scientific_blockers
    assert report.benchmark_outcomes_read is False
    assert report.utility_read is False


if __name__ == "__main__":
    test_empirical_gate_requires_determinism_extinction_recall_and_zero_false_veto()
    print("[semantic-coverage-empirical-audit-v1] implementation gate: pass")
