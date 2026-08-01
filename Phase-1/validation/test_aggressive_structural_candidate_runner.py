#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggressive_structural_candidate_runner import run_candidate_arms


def main() -> int:
    repeated = "This repeated transport boilerplate is long enough to be compacted while each chunk keeps independent payload."
    header = "# SPDX-License-Identifier: Apache-2.0\n# Licensed under the Apache License, Version 2.0"
    base = " ".join(f"token{index}" for index in range(100))
    rows = [
        {"chunk_uid": "a", "stage_a_record_id": "a", "text": f"{header}\n\ndef alpha():\n    return 1\n\n{repeated}", "token_proxy": 30},
        {"chunk_uid": "b", "stage_a_record_id": "b", "text": f"{header}\n\ndef beta():\n    return 2\n\n{repeated}", "token_proxy": 30},
        {"chunk_uid": "c", "stage_a_record_id": "c", "text": base, "token_proxy": 100},
        {"chunk_uid": "d", "stage_a_record_id": "d", "text": base.replace("token99", "replacement"), "token_proxy": 100},
    ]
    result = run_candidate_arms(
        rows,
        stage_c_selection={
            "near_duplicate_compaction": {
                "candidate_enabled": True,
                "shingle_size": 5,
                "minimum_lexical_tokens": 40,
                "symmetric_overlap_threshold": 0.95,
            }
        },
        minimum_chunk_chars=30,
        token_counter=lambda text: len(text.split()),
    )

    assert result["runtime_active"] is False
    assert set(result["arms"]) == {
        "active_profile_baseline",
        "license_span_compaction",
        "repeated_span_compaction",
        "strengthened_duplicate_family",
        "cumulative_aggressive_candidate",
    }
    assert result["arms"]["active_profile_baseline"]["summary"]["transformed_span_count"] == 0
    assert result["arms"]["license_span_compaction"]["summary"]["transformed_span_count"] == 2
    assert result["arms"]["repeated_span_compaction"]["summary"]["transformed_span_count"] == 1
    assert result["arms"]["active_profile_baseline"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is False
    assert result["arms"]["license_span_compaction"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is False
    assert result["arms"]["repeated_span_compaction"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is False
    assert result["arms"]["strengthened_duplicate_family"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is True
    assert result["arms"]["cumulative_aggressive_candidate"]["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is True
    assert result["arms"]["strengthened_duplicate_family"]["stage_c_selection"]["near_duplicate_compaction"]["symmetric_overlap_threshold"] == 0.9
    assert result["arms"]["cumulative_aggressive_candidate"]["summary"]["curated_token_count"] < result["arms"]["active_profile_baseline"]["summary"]["curated_token_count"]
    assert set(result["near_duplicate_threshold_sweep"]) == {"0.90", "0.92", "0.95"}
    assert all(
        arm["stage_c_selection"]["near_duplicate_compaction"]["candidate_enabled"] is True
        for arm in result["near_duplicate_threshold_sweep"].values()
    )
    assert result["near_duplicate_threshold_sweep"]["0.95"]["summary"]["curated_token_count"] >= result["near_duplicate_threshold_sweep"]["0.90"]["summary"]["curated_token_count"]
    for arm in result["arms"].values():
        assert arm["selector_boundary"]["source_identity_read"] is False
        assert arm["selector_boundary"]["benchmark_outcomes_read"] is False

    print("[aggressive-structural-candidate-runner] five-arm isolated candidate materialization: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
