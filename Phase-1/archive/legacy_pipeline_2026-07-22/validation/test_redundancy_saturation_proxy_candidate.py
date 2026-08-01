#!/usr/bin/env python3
"""Validate the frozen saturation proxy-training candidate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = json.loads(
        (
            ROOT
            / "configs"
            / "temporal_code_redundancy_saturation_proxy_candidate_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "frozen_before_proxy_model_outcomes"
    assert payload["candidate"] == "log_count"
    assert payload["canonical_control"] == "binary_current"
    assert payload["selection_rationale"]["outcome_free"] is True
    assert payload["proxy_training_arms"] == [
        "binary_current",
        "log_count",
        "stageA_random_equal_budget",
    ]
    assert "confirmatory outcomes" in payload["forbidden_inputs"]
    print("[redundancy-saturation-proxy] log-count candidate frozen without outcome leakage: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
