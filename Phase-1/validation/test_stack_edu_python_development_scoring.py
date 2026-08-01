#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_stack_edu_python_development import build_score_row


def test_provider_runs_only_after_python_complete_source_structural_gate() -> None:
    # Given: one supported implementation and one language-unknown record.
    scored_texts: list[str] = []

    def scorer(text: str) -> float:
        scored_texts.append(text)
        return 3.25

    supported = {
        "record_id": "python-file",
        "text": "def add(a, b):\n    return a + b\n",
        "language": {"code": "python", "declaration": "source_row"},
        "record_shape": "complete_source",
    }
    unsupported = {
        "record_id": "unknown-file",
        "text": "def add(a, b):\n    return a + b\n",
        "language": {"code": "und"},
        "record_shape": "complete_source",
    }

    # When: frozen provider score rows are built.
    supported_result = build_score_row(supported, scorer, "provider-revision")
    unsupported_result = build_score_row(unsupported, scorer, "provider-revision")

    # Then: only the fully in-scope record consumes the provider.
    assert supported_result["route_specific_evidence"] == 3.25
    assert supported_result["structural_heads"] == {
        "route_confidence": 1.0,
        "substantive_payload": 1.0,
        "coherence_completeness": 1.0,
    }
    assert unsupported_result["route_specific_evidence"] is None
    assert len(scored_texts) == 1


if __name__ == "__main__":
    test_provider_runs_only_after_python_complete_source_structural_gate()
    print("[stack-edu-python-scoring] scoped provider invocation: pass")
