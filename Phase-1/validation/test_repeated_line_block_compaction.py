#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repeated_line_block_compaction import build_plan, materialize_candidate_plan


def test_repeated_navigation_block_compacts_only_the_second_occurrence() -> None:
    block = "Home\nGallery\nContact"
    text = f"Archive article body contains the actual historical description.\n\n{block}\n\nMore article content remains available to readers.\n\n{block}"
    rows = [{"chunk_uid": "nav", "text": text}]
    plan = build_plan(rows, minimum_residual_chars=40)
    result = materialize_candidate_plan(rows, plan)

    assert plan["candidate_span_removals"] == 1
    proposal = plan["proposals"][0]
    assert proposal["representative_occurrence"] == "earlier_in_same_chunk"
    assert proposal["representative_block_sha256"]
    assert result["records"][0]["text"].count(block) == 1


def test_tables_code_quotes_and_sentence_prose_are_retained() -> None:
    rows = [
        {"chunk_uid": "table", "text": "Name | Value | Count\nName | Value | Count\nName | Value | Count"},
        {"chunk_uid": "code", "text": "import alpha\nimport beta\nimport gamma\nimport alpha\nimport beta\nimport gamma"},
        {"chunk_uid": "prose", "text": "This sentence explains a result.\nThis sentence explains a result.\nThis sentence explains a result."},
    ]
    plan = build_plan(rows, minimum_residual_chars=20)

    assert plan["candidate_span_removals"] == 0


def test_repeated_headings_references_and_test_matrix_are_retained() -> None:
    rows = [
        {
            "chunk_uid": "headings",
            "text": "Introduction\nMethod\nResults\nThe first section explains the study.\n\nIntroduction\nMethod\nResults\nThe appendix gives the full methodology.",
        },
        {
            "chunk_uid": "references",
            "text": "Smith, A. (2024). Study one.\nSmith, A. (2024). Study one.\nSmith, A. (2024). Study one.",
        },
        {
            "chunk_uid": "matrix",
            "text": "Case 1: pass\nCase 2: fail\nCase 3: pass\nCase 1: pass\nCase 2: fail\nCase 3: pass",
        },
    ]

    plan = build_plan(rows, minimum_residual_chars=20)

    assert plan["candidate_span_removals"] == 0


if __name__ == "__main__":
    test_repeated_navigation_block_compacts_only_the_second_occurrence()
    test_tables_code_quotes_and_sentence_prose_are_retained()
    test_repeated_headings_references_and_test_matrix_are_retained()
    print("[repeated-line-block] candidate boundary: pass")
