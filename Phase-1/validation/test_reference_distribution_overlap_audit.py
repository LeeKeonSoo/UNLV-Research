#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_reference_distribution_overlap import build_overlap_audit


def test_overlap_audit_reports_existing_stage_a_evidence_without_removal() -> None:
    review_sample = {"review_records": [{"record_id": "raw-1"}, {"record_id": "raw-2"}]}
    candidates = [
        {
            "record_id": "raw-1",
            "language": {"code": "python"},
            "release_eligibility": {"eligible": True},
            "rights": {"status": "allowed"},
            "quarantine": {"status": "release_candidate"},
            "hazards": {"pii_detected": False, "secret_detected": False, "poisoning_suspected": False, "benchmark_contamination": False},
            "composition": {"content_domain": "code"},
        },
        {
            "record_id": "raw-2",
            "language": {"code": "python"},
            "release_eligibility": {"eligible": False},
            "rights": {"status": "unknown"},
            "quarantine": {"status": "quarantined"},
            "hazards": {"pii_detected": True},
            "composition": {"content_domain": "code"},
        },
    ]

    report = build_overlap_audit(review_sample, candidates)

    assert report["status"] == "overlap_audit_complete_not_a_selection_policy"
    assert report["summary"]["all_stage_a_evidence_present"] == 1
    assert report["summary"]["needs_review"] == 1
    assert report["records"][0]["evidence_status"] == "eligible_code_overlap_not_a_removal_decision"
    assert report["selector_boundary"]["data_removed"] is False


if __name__ == "__main__":
    test_overlap_audit_reports_existing_stage_a_evidence_without_removal()
    print("[reference-distribution-overlap-audit] Stage-A overlap boundary: pass")
