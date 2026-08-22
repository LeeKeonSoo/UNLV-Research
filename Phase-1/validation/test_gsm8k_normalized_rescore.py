#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.gsm8k_normalized_rescore import (
    RescoreSource,
    build_rescore_artifact,
    score_response,
)


def main() -> int:
    # Given: a correct answer followed by an over-generated second problem.
    response = (
        "The calculation gives 18. The answer is 18.\n"
        "Question: A new problem contains 1,000 and 140."
    )
    # When: the response is normalized against the GSM8K target.
    score = score_response(response, "work\n#### 18")
    # Then: only the first answer segment controls correctness.
    assert score.correct
    assert score.truncated_next_problem
    assert score.extraction_method == "explicit_final_answer"

    # Given: equivalent currency, comma, decimal, fraction, and fallback forms.
    cases = (
        ("Therefore, the final answer is $1,234.", "#### 1234", True),
        ("The result is \\boxed{0.5}.", "#### 1/2", True),
        ("Six groups of seven give 42", "#### 42", True),
        ("The answer is 41.", "#### 42", False),
    )
    # When/Then: numeric normalization is deterministic for every form.
    for generated, target, expected in cases:
        assert score_response(generated, target).correct is expected

    # Given: one response duplicated across the official strict/flexible filters.
    samples = [
        {
            "doc_id": 0,
            "target": "steps\n#### 18",
            "resps": [[response]],
            "filter": "strict-match",
            "exact_match": 1.0,
        },
        {
            "doc_id": 0,
            "target": "steps\n#### 18",
            "resps": [[response]],
            "filter": "flexible-extract",
            "exact_match": 0.0,
        },
        {
            "doc_id": 1,
            "target": "steps\n#### 7",
            "resps": [["No numeric answer was produced."]],
            "filter": "strict-match",
            "exact_match": 0.0,
        },
        {
            "doc_id": 1,
            "target": "steps\n#### 7",
            "resps": [["No numeric answer was produced."]],
            "filter": "flexible-extract",
            "exact_match": 0.0,
        },
    ]
    payload = {
        "results": {
            "gsm8k_cot_zeroshot": {
                "exact_match,strict-match": 0.5,
                "exact_match,flexible-extract": 0.0,
            }
        },
        "samples": {"gsm8k_cot_zeroshot": samples},
    }
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / "result.json"
        source_path.write_text(json.dumps(payload), encoding="utf-8")
        # When: the complete artifact is built from the official result file.
        artifact = build_rescore_artifact((RescoreSource("fixture", source_path),))
    # Then: filter duplicates are collapsed and audit counts remain attributable.
    result = artifact.results[0]
    assert result.records == 2
    assert result.correct == 1
    assert result.unparsed == 1
    assert result.truncated_next_problem == 1
    assert result.official_strict_accuracy == 0.5
    assert result.official_flexible_accuracy == 0.0
    print("[gsm8k-normalized-rescore] parser and artifact contracts: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
