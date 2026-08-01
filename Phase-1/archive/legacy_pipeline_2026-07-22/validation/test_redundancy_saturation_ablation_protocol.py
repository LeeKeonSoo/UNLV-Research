#!/usr/bin/env python3
"""Validate the frozen Redundancy saturation ablation protocol."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = json.loads(
        (ROOT / "configs" / "temporal_code_redundancy_saturation_ablation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "frozen_before_target_model_outcomes"
    assert payload["common_contract"]["utility_consumed"] is False
    assert payload["common_contract"]["benchmark_outcomes_consumed"] is False
    assert set(payload["arms"]) == {"binary_current", "exp_tau_1", "exp_tau_2", "log_count"}
    assert payload["common_contract"]["full_curated_pool_retained"] is True
    assert payload["promotion_gate"]
    print("[redundancy-saturation-ablation] outcome-free arms and promotion gate: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
