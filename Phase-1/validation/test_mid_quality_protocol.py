#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "mid_quality_protocol_v1.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["schema_version"] == "mid-quality-protocol-v1"
    assert protocol["status"] == "preregistered_not_runtime_active"
    assert protocol["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert protocol["base_model"] == {
        "model_id": "Qwen/Qwen3-4B-Base",
        "snapshot_id": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        "config_sha256": "304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1",
        "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    }
    assert protocol["development_split"]["corpora"] == ["code_raw_like", "math_raw_like", "general_text_raw_like"]
    assert protocol["development_split"]["disjointness_keys"] == ["stable_record_id", "normalized_text_sha256"]
    assert protocol["development_target"]["kind"] == "heldout_continuation_loss"
    assert protocol["development_target"]["benchmark_disjoint"] is True
    assert protocol["estimator"]["unit"] == "semantic_or_structural_group"
    assert protocol["estimator"]["implementation"] == "mid_quality_estimator.build_mid_quality_development_report"
    assert protocol["estimator"]["report_schema_version"] == "mid-quality-development-report-v1"
    assert protocol["estimator"]["calibration"] == "null_control_bootstrap_margin"
    assert protocol["estimator"]["selection_rule"] == "negative_only_after_calibrated_upper_confidence_bound_is_non_positive"
    assert protocol["estimator"]["proxy_nll_rank_is_not_an_estimator"] is True
    assert protocol["training_recipe"]["method"] == "QLoRA_continued_pretraining"
    assert protocol["forbidden_runtime_inputs"] == ["Quality", "Utility", "NLL", "benchmark_outcomes", "source_identity", "domain", "target_retention_fraction", "budget"]
    assert protocol["existing_stage_c2_candidate"]["decision"] == "remains_archived_not_a_mid_estimator"
    assert "known_high_quality_reference_false_positive_risk" in protocol["existing_stage_c2_candidate"]["blocking_evidence"]

    print("[mid-quality-protocol] frozen model, split, target, and estimator boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
