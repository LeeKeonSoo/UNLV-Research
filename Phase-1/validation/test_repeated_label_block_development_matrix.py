#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repeated_label_block_development_matrix import run_candidate_matrix


def test_candidate_matrix_reports_token_delta_and_preservation_invariants() -> None:
    block = "Home\nGallery\nContact"
    rows = [
        {
            "chunk_uid": "navigation",
            "text": f"The archive contains a substantive article body.\n\n{block}\n\nA second substantive paragraph remains.\n\n{block}",
        },
        {"chunk_uid": "heading", "text": "Introduction\nMethod\nResults\n\nThis article explains the method."},
    ]

    report = run_candidate_matrix(rows, minimum_residual_chars=40, token_counter=lambda text: len(text.split()))

    assert report["status"] == "development_candidate_complete_not_runtime_active"
    assert report["runtime_active"] is False
    assert report["summary"]["candidate_span_removals"] == 1
    assert report["summary"]["token_delta"] < 0
    assert report["coverage"]["first_occurrence_linkage_passed"] is True
    assert report["coverage"]["residual_payload_passed"] is True
    assert report["coverage"]["whole_chunk_preservation_passed"] is True
    reasons = report["reason_code_impact_audit"]["stages"]["stage_c_span_transformation"]["reasons"]
    assert reasons["repeated_label_block_removed"]["chunks"] == 1


if __name__ == "__main__":
    test_candidate_matrix_reports_token_delta_and_preservation_invariants()
    print("[repeated-label-block-development-matrix] candidate matrix: pass")
