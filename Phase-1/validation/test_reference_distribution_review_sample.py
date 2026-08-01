#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_reference_distribution_review_sample import build_review_sample


def test_review_sample_is_ranked_but_not_a_selection_decision() -> None:
    reference_train = [
        {"record_id": "reference-1", "text": "def parse_contract(value):\n    return validate_schema(value)"},
        {"record_id": "reference-2", "text": "class StableClient:\n    def request(self, payload):\n        return serialize(payload)"},
    ]
    candidates = [
        {"record_id": "raw-1", "text": "def validate_contract(value):\n    return parse_schema(value)", "partition": {"source_tier": "raw_like", "source_dataset": "raw", "repository_identity": "org/a", "path": "src/a.py"}},
        {"record_id": "raw-2", "text": "temporary generated log output", "partition": {"source_tier": "raw_like", "source_dataset": "raw", "repository_identity": "org/b", "path": "src/b.py"}},
        {"record_id": "raw-3", "text": "def parse_schema(value):\n    return validate_contract(value)", "partition": {"source_tier": "raw_like", "source_dataset": "raw", "repository_identity": "org/c", "path": "src/c.py"}},
    ]

    report = build_review_sample(reference_train, candidates, split_salt="fixture", sample_size=1)

    assert report["status"] == "review_sample_ready_labels_required_not_a_selection_policy"
    assert report["summary"]["review_records"] == 1
    assert report["review_records"][0]["label_status"] == "unlabeled"
    assert report["review_records"][0]["repository"].startswith("org/")
    assert report["review_records"][0]["path"].startswith("src/")
    assert report["selector_boundary"]["data_removed"] is False
    assert report["selector_boundary"]["selection_decisions_emitted"] is False


if __name__ == "__main__":
    test_review_sample_is_ranked_but_not_a_selection_decision()
    print("[reference-distribution-review-sample] review-only boundary: pass")
