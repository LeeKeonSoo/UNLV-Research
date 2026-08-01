#!/usr/bin/env python3
"""Contract checks for broad Stage-B frozen selector ablations."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    path = PROJECT_DIR / "outputs" / "validation" / "temporal_code_broad_stage_b_ablations.json"
    if not path.exists():
        print("[temporal-code-broad-stage-b-ablations] generated report absent; contract test skipped")
        return 0
    report = load_json(path)
    assert report["schema_version"] == "temporal-code-broad-stage-b-ablations-v1"
    assert set(report["arms"]) == {"full_selector", "quality_only", "redundancy_only", "no_coverage_support"}
    assert report["forbidden_signals_observed"] == []
    assert report["raw_random_equal_token"]["status"] == "pending_target_tokenizer_construction"
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-broad-stage-b-ablations] frozen proxy arms and no-leak contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
