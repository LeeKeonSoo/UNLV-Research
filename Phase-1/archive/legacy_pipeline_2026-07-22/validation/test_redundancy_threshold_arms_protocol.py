#!/usr/bin/env python3
"""Validate frozen hard-near-duplicate threshold arms."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = json.loads(
        (ROOT / "configs" / "temporal_code_hard_near_duplicate_threshold_arms_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "frozen_after_development_calibration_before_holdout_evaluation"
    assert payload["current_arm"] == "current"
    assert len(payload["arms"]) == 5
    assert "Utility" in payload["forbidden_inputs"]
    assert payload["selection_rule_after_holdout"]
    print("[redundancy-threshold-arms] frozen before holdout with non-Utility selection rule: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
