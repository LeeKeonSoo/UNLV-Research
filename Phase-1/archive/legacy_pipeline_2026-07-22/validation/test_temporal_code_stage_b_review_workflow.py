#!/usr/bin/env python3
"""Regression checks for blind-review entry, freeze, and adjudication workflow."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _packet(reviewer: str, labels: list[tuple[str, str] | None]) -> dict:
    records = []
    for index, label in enumerate(labels):
        quality, redundancy = label if label else (None, None)
        records.append(
            {
                "review_id": f"r-{index}",
                "text": f"record {index}",
                "review_fields": {
                    "quality_label": quality,
                    "redundancy_label": redundancy,
                    "confidence": "high" if label else None,
                    "notes": None,
                },
            }
        )
    return {"status": "awaiting_independent_review", "reviewer_id": reviewer, "records": records}


def main() -> int:
    module = importlib.import_module("84_manage_temporal_code_stage_b_review")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        a_path, b_path, adj_path = root / "a.json", root / "b.json", root / "adj.json"
        a_path.write_text(json.dumps(_packet("a", [None, ("preserve", "unique")])), encoding="utf-8")
        b_path.write_text(json.dumps(_packet("b", [("downrank", "unique"), ("preserve", "unique")])), encoding="utf-8")
        adj_path.write_text(
            json.dumps(
                {
                    "status": "inactive_until_two_independent_reviews_are_frozen",
                    "records": [
                        {"review_id": f"r-{i}", "text": f"record {i}", "adjudication_fields": {"quality_label": None, "redundancy_label": None, "confidence": None, "notes": None}}
                        for i in range(2)
                    ],
                }
            ),
            encoding="utf-8",
        )
        try:
            module.freeze_review(a_path, "reviewer-a", attest_independent=True, attest_no_key=True)
            raise AssertionError("Incomplete review freeze should fail.")
        except RuntimeError:
            pass
        shown = module.show_record(load_json(a_path))
        assert shown["review_id"] == "r-0", shown
        assert "stage_b_evidence" not in shown, shown
        module.set_label(a_path, "r-0", "preserve", "unique", "high", None)
        module.freeze_review(a_path, "reviewer-a", attest_independent=True, attest_no_key=True)
        module.freeze_review(b_path, "reviewer-b", attest_independent=True, attest_no_key=True)
        activated = module.activate_adjudication(a_path, b_path, adj_path)
        assert activated["record_count"] == 1, activated
        assert load_json(adj_path)["records"][0]["review_id"] == "r-0"
    print("[temporal-code-stage-b-review-workflow] entry, freeze, and disagreement activation: pass")
    return 0


if __name__ == "__main__":
    from data_eval_common import load_json

    raise SystemExit(main())
