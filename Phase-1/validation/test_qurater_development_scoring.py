#!/usr/bin/env python3
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_qurater_development import (
    ProviderScoreError,
    QuRaterScores,
    aggregate_window_scores,
    score_metadata,
)


def test_window_scores_use_model_card_token_weighting() -> None:
    scores = aggregate_window_scores(
        (
            QuRaterScores(1.0, 2.0, 3.0, 4.0),
            QuRaterScores(3.0, 4.0, 5.0, 6.0),
        ),
        (3, 1),
    )

    assert scores == QuRaterScores(1.5, 2.5, 3.5, 4.5)


def test_window_aggregation_preserves_unbounded_finite_logits() -> None:
    scores = aggregate_window_scores((QuRaterScores(-2.0, 0.0, 4.5, 8.0),), (12,))

    assert scores.educational_value == 8.0
    assert scores.writing_style == -2.0


def test_window_aggregation_rejects_empty_or_non_finite_evidence() -> None:
    for windows, counts in (
        ((), ()),
        ((QuRaterScores(math.nan, 0.0, 0.0, 0.0),), (1,)),
        ((QuRaterScores(0.0, 0.0, 0.0, 0.0),), (0,)),
    ):
        try:
            aggregate_window_scores(windows, counts)
        except ProviderScoreError:
            continue
        raise AssertionError("Invalid provider evidence must fail closed.")


def test_score_metadata_is_text_only_stable_and_route_scoped() -> None:
    prose = (
        "This explanation contains a complete account of the experiment and its evidence. "
        "It therefore gives readers enough information to understand the result."
    )

    first = score_metadata(prose)
    second = score_metadata(f"  {prose}\r\n")

    assert first["normalized_text_sha256"] == second["normalized_text_sha256"]
    assert first["general_informational_prose"] is True
    assert score_metadata("https://example.com\nhttps://example.org\nhttps://example.net")[
        "general_informational_prose"
    ] is False


def test_scoring_cli_can_resolve_repository_modules() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "score_qurater_development.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    test_window_scores_use_model_card_token_weighting()
    test_window_aggregation_preserves_unbounded_finite_logits()
    test_window_aggregation_rejects_empty_or_non_finite_evidence()
    test_score_metadata_is_text_only_stable_and_route_scoped()
    test_scoring_cli_can_resolve_repository_modules()
    print("[qurater-development-scoring] native logits and window weighting: pass")
