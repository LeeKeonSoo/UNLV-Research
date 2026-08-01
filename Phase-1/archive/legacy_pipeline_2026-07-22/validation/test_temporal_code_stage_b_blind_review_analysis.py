#!/usr/bin/env python3
"""Regression checks for temporal-code Stage-B blind-review analysis gates."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> int:
    module = importlib.import_module("79_analyze_temporal_code_stage_b_blind_review")
    packet = {
        "records": [
            {
                "review_id": "review-a",
                "review_fields": {"quality_label": None, "redundancy_label": None, "confidence": None},
            }
        ]
    }
    key = {
        "records": [
            {
                "review_id": "review-a",
                "repository_identity": "fixture/repo",
                "stage_b_evidence": {"code_quality_proxy": 0.9, "soft_redundancy_risk": 0.1},
            }
        ]
    }
    blocked = module.analyze(packet, key)
    assert blocked["status"] == "blocked_incomplete_independent_review", blocked
    assert blocked["proxy_promotion_allowed"] is False, blocked
    packet["records"][0]["review_fields"] = {
        "quality_label": "preserve",
        "redundancy_label": "unique",
        "confidence": "high",
    }
    complete = module.analyze(packet, key)
    assert complete["status"] == "independent_review_complete_initial_real_corpus_evidence", complete
    assert complete["proxy_promotion_allowed"] is False, complete
    print("[temporal-code-stage-b-review-analysis] incomplete review blocks analysis: pass")
    print("[temporal-code-stage-b-review-analysis] complete single-repo review cannot promote proxy: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
