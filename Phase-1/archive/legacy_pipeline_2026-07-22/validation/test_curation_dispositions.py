#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.curation_dispositions import annotate_retained_pool


def test_budget_exclusion_preserves_curation_without_quality_label() -> None:
    rows = annotate_retained_pool(
        [{"chunk_uid": "one", "text": "usable text"}],
        selected_ids=set(),
        budget_applied=True,
    )
    decision = rows[0]["curation_decision"]
    assert decision["curation_disposition"] == "retained"
    assert decision["training_budget_disposition"] == "budget_not_selected"
    assert decision["budget_exclusion_is_rejection"] is False
    assert "quality_judgment" not in decision


if __name__ == "__main__":
    test_budget_exclusion_preserves_curation_without_quality_label()
    print("[policy-dispositions] budget exclusion preserves curation: pass")
