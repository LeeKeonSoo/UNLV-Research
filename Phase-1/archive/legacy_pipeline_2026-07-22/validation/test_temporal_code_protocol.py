#!/usr/bin/env python3
"""Regression checks for the temporal code curation preregistration."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def _date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json")
    assert protocol["schema_version"] == "temporal-code-curation-protocol-v1"
    assert protocol["status"] == "preregistered_before_collection"
    assert protocol["models"]["primary"]["model_id"] == "Qwen/Qwen3-4B-Base"
    assert protocol["models"]["primary"]["training_stage"] == "pretraining"
    assert protocol["claim_scope"]["primary"].startswith("parameter-efficient continued pretraining")

    collection = protocol["collection_contract"]
    assert _date(collection["training_window"]["start"]) >= _date(protocol["models"]["primary"]["safe_collection_start"])
    assert _date(collection["training_window"]["end"]) < _date(collection["development_holdout_window"]["start"])
    assert _date(collection["development_holdout_window"]["end"]) < _date(
        collection["frozen_confirmatory_holdout_window"]["start"]
    )
    assert protocol["split_contract"]["repository_identity_cannot_cross_splits"] is True
    assert protocol["split_contract"]["split_assignment_frozen_before_core_scoring"] is True
    assert "excluded from training" in collection["training_payload_rule"]
    assert any("rate limits" in rule for rule in collection["acquisition_rules"])

    assert "project-created temporal executable holdouts" in protocol["benchmark_quarantine"]["never_training_sources"]
    assert protocol["training_contract"]["minimum_development_seeds"] >= 5
    assert protocol["training_contract"]["minimum_fresh_confirmatory_seeds"] >= 5
    assert "stageA_random_equal_token" in protocol["comparison_arms"]
    stage_b = protocol["stage_b_contract"]
    assert stage_b["input"] == "train split Stage-A-pass chunks only"
    assert stage_b["coverage_support"]["role"].startswith("selection constraint only")
    assert stage_b["coverage_support"]["minimum_relative_token_share"] == 0.5
    assert "Utility" in stage_b["objective"]["forbidden_signals"]
    assert stage_b["objective"]["redundancy_search_mode"] == "indexed_exact"
    assert stage_b["objective"]["indexed_search_must_match_all_pairs_on_smoke"] is True
    assert stage_b["stage_a_random_baseline"]["must_be_disjoint_from_selected"] is True
    philosophy = protocol["validation_philosophy"]
    assert philosophy["human_or_llm_review"].startswith("optional diagnostic only")
    assert "equal-budget downstream Stage-C comparison" in philosophy["primary_stage_b_validation"]
    assert len(philosophy["project_specific_parameters"]) >= 4
    assert "not validated by citation alone" in philosophy["parameter_claim"]
    assert protocol["future_release_rule"]["task_requirement"] == "NLL-only evidence cannot support release"
    assert protocol["utility_scope"] == "Stage C validation only; never selector objective"
    print("[temporal-code-protocol] model, temporal split, contamination, and Stage-C contracts: pass")
    print("[temporal-code-protocol] automated primary validation and optional-review boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
