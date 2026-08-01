#!/usr/bin/env python3
"""Contract checks for temporal-code Qwen3-4B Stage-C smoke."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    contract = load_json(PROJECT_DIR / "configs" / "temporal_code_stage_c_smoke_qwen3_4b_v1.json")
    assert contract["status"] == "frozen_before_target_tokenization_or_model_execution"
    assert contract["target_model"]["model_id"] == "Qwen/Qwen3-4B-Base"
    arm_contract = contract["arm_contract"]
    assert arm_contract["all_sensitivity_arms_must_share_common_stage_a_baseline"] is True
    assert arm_contract["common_stage_a_baseline_must_be_disjoint_from_every_sensitivity_arm"] is True
    assert contract["utility_scope"] == "Stage C validation only; never selector objective"
    assert contract["human_or_llm_review"].startswith("optional diagnostic only")
    manifest_path = (
        PROJECT_DIR
        / "outputs"
        / "temporal_code_stage_c_smoke_qwen3_4b_v1"
        / "frozen_smoke_arm_manifest.json"
    )
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        assert manifest["curated_common_baseline_overlap_count"] == 0
        assert manifest["all_sensitivity_arms_share_common_stage_a_baseline"] is True
        assert manifest["confirmatory_content_read"] is False
        assert len(set(manifest["sensitivity_common_stage_a_baseline_sha256"].values())) == 1
    print("[temporal-code-stage-c-smoke] frozen common-baseline and no-leak contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
