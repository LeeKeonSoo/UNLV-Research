#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_math_positive_development import build_score_row, predict_positive_probability


def test_row_metadata_cannot_replace_text_only_provider_scores() -> None:
    row = {
        "record_id": "math-page",
        "text": "Let x be a real number. We prove x squared is nonnegative.",
        "token_count": 17,
        "partition": {"source_row_metadata": '{"math_score": 1.0}'},
    }

    result = build_score_row(row, 0.61, False, 3.25, "math-rev", "fine-rev")

    assert result["route_confidence"] == 0.61
    assert result["route_confidence_evidence"] == {
        "explicit_math_notation": False,
        "mathscore_probability": 0.61,
    }
    assert result["route_specific_evidence"] == 3.25
    assert result["missing_heads"] == ["substantive_payload", "coherence_completeness"]
    assert result["decision"] == "abstain"
    assert "partition" not in result


def test_fasttext_adapter_uses_native_predictions_without_numpy_copy_contract() -> None:
    class Native:
        def predict(self, text: str, k: int, threshold: float, on_unicode_error: str):
            assert text.endswith("\n")
            return [(0.8, "__label__positive"), (0.2, "__label__negative")]

    class Model:
        f = Native()

    assert predict_positive_probability(Model(), "math text") == 0.8


if __name__ == "__main__":
    test_row_metadata_cannot_replace_text_only_provider_scores()
    test_fasttext_adapter_uses_native_predictions_without_numpy_copy_contract()
    print("[math-positive-scoring] text-only provider boundary: pass")
