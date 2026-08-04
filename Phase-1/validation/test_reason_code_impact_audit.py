#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reason_code_audit import build_reason_code_impact_audit


def test_reason_code_impact_audit_reports_stage_specific_record_chunk_and_token_cost() -> None:
    report = build_reason_code_impact_audit(
        stage_a_quarantined=[
            {
                "record_id": "a1",
                "text": "one two three",
                "quarantine": {"reasons": ["pii_detected", "rights_unknown"]},
            }
        ],
        stage_b_rejected=[
            {
                "stage_a_record_id": "b1",
                "token_proxy": 7,
                "stage_b_hard_gate_reasons": ["normalized_exact_duplicate"],
            }
        ],
        stage_b_not_selected=[
            {
                "stage_a_record_id": "c1",
                "token_proxy": 5,
                "stage_b_policy": {"removed_reason": "near_duplicate_representative_retained"},
            }
        ],
        stage_b_transformations=[
            {
                "chunk_uid": "c2",
                "reason_code": "repeated_exact_template_span_removed",
                "span_token_proxy": 12,
                "pre_token_proxy": 25,
                "post_token_proxy": 7,
            }
            ,
            {
                "chunk_uid": "c3",
                "reason_code": "inline_license_header_removed",
                "header_token_proxy": 10,
            }
        ],
    )

    assert report["schema_version"] == "reason-code-impact-audit-v1"
    assert report["authority"] == "audit_only"
    assert report["stages"]["stage_a_quarantine"]["reasons"]["pii_detected"] == {
        "records": 1,
        "chunks": 1,
        "token_proxy": 3,
    }
    assert report["stages"]["stage_b_rejection"]["reasons"]["normalized_exact_duplicate"]["token_proxy"] == 7
    assert report["stages"]["stage_b_policy_removal"]["reasons"]["near_duplicate_representative_retained"]["records"] == 1
    assert report["stages"]["stage_b_span_transformation"]["reasons"]["repeated_exact_template_span_removed"] == {
        "chunks": 1,
        "token_proxy_removed": 12,
    }
    assert report["stages"]["stage_b_span_transformation"]["reasons"]["inline_license_header_removed"] == {
        "chunks": 1,
        "token_proxy_removed": 10,
    }


if __name__ == "__main__":
    test_reason_code_impact_audit_reports_stage_specific_record_chunk_and_token_cost()
    print("[reason-code-impact-audit] stage reason-code impact accounting: pass")
