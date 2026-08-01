#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_general_provider_candidates_v2 import (
    dclm_high_quality_probability,
    fineweb_regression_score,
    normalize_fasttext_document,
    predict_dclm_labels,
)


def test_dclm_probability_is_selected_by_label_not_position() -> None:
    labels = ("__label__cc", "__label__hq")
    probabilities = (0.91, 0.09)

    assert dclm_high_quality_probability(labels, probabilities) == 0.09


def test_dclm_document_is_one_normalized_fasttext_line() -> None:
    text = "First line.\n\nSecond\tline."

    assert normalize_fasttext_document(text) == "First line. Second line."


def test_dclm_prediction_uses_numpy2_compatible_native_result() -> None:
    class NativeModel:
        @staticmethod
        def predict(text: str, k: int, threshold: float, on_unicode_error: str):
            assert text.endswith("\n")
            return [(0.8, "__label__hq"), (0.2, "__label__cc")]

    class Model:
        f = NativeModel()

    labels, probabilities = predict_dclm_labels(Model(), "one line")

    assert labels == ("__label__hq", "__label__cc")
    assert probabilities == (0.8, 0.2)


def test_fineweb_regression_requires_one_finite_logit() -> None:
    assert fineweb_regression_score((2.75,)) == 2.75

    try:
        fineweb_regression_score((1.0, 2.0))
    except ValueError as error:
        assert "one regression logit" in str(error)
    else:
        raise AssertionError("multiple logits must be rejected")


if __name__ == "__main__":
    test_dclm_probability_is_selected_by_label_not_position()
    test_dclm_document_is_one_normalized_fasttext_line()
    test_dclm_prediction_uses_numpy2_compatible_native_result()
    test_fineweb_regression_requires_one_finite_logit()
    print("general provider candidate scoring v2: ok")
