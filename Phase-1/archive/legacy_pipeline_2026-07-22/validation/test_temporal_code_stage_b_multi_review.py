#!/usr/bin/env python3
"""Regression checks for independent multi-review packets and analysis gates."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _label(packet: dict, quality: str, redundancy: str) -> None:
    for row in packet["records"]:
        fields = row.get("review_fields") or row.get("adjudication_fields")
        fields.update({"quality_label": quality, "redundancy_label": redundancy, "confidence": "high"})


def main() -> int:
    builder = importlib.import_module("82_build_temporal_code_stage_b_multi_reviewer_packets")
    analyzer = importlib.import_module("83_analyze_temporal_code_stage_b_multi_review")
    master = {
        "records": [
            {"review_id": f"r-{i}", "content_type": "code", "change_type": "modified", "chunk_kind": "function", "text": f"def f{i}(): pass", "review_fields": {}}
            for i in range(4)
        ]
    }
    key = {
        "records": [
            {"review_id": f"r-{i}", "repository_identity": f"repo/{i % 3}", "stage_b_evidence": {"code_quality_proxy": 0.9, "soft_redundancy_risk": 0.1}}
            for i in range(4)
        ]
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        builder.build(master, root)
        from data_eval_common import load_json
        a = load_json(root / "reviewer_a_packet.json")
        b = load_json(root / "reviewer_b_packet.json")
        adj = load_json(root / "adjudication_packet.json")
    forbidden = {"repository_identity", "path", "arm", "stage_b_evidence", "sampling_stratum"}
    assert all(not forbidden.intersection(row) for row in a["records"]), a
    assert [row["review_id"] for row in a["records"]] != [row["review_id"] for row in b["records"]], (a, b)
    blocked = analyzer.analyze(a, b, adj, key)
    assert blocked["status"] == "blocked_incomplete_independent_reviews", blocked
    _label(a, "preserve", "unique")
    _label(b, "preserve", "unique")
    complete = analyzer.analyze(a, b, adj, key)
    assert complete["status"] == "multi_review_complete_initial_real_corpus_evidence", complete
    assert complete["inter_reviewer_agreement"]["quality_cohen_kappa"] == 1.0, complete
    assert complete["proxy_promotion_allowed"] is False, complete
    print("[temporal-code-stage-b-multi-review] independent packets and gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
