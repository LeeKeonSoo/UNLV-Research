#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "mode_development_ablation_protocol_v1.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "mode-development-ablation-protocol-v1"
    assert protocol["status"] == "preregistered_candidate_only_not_runtime_active"
    assert protocol["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert protocol["development_matrix"] == ["code_raw_like", "math_raw_like", "general_text_raw_like"]
    assert protocol["arms"] == ["weak", "mid", "hard"]
    assert protocol["required_artifacts"] == ["frozen_weak_stage_c_jsonl", "frozen_group_membership_manifest", "mid_quality_development_report", "hard_quality_candidate_plan"]
    assert protocol["external_evaluation"]["training_budget"] == "natural_budget_per_materialized_arm"
    assert protocol["external_evaluation"]["feedback_into_policy"] is False
    assert "legacy_stage_c2_proxy_artifacts" in protocol["excluded_inputs"]

    print("[mode-development-ablation-protocol] frozen three-arm development boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
