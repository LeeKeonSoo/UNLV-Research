from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "configs" / "code_5m_external_validation_study_v1.json"
EXECUTION = ROOT / "configs" / "code_5m_natural_budget_execution_qwen3_4b_v1.json"


def test_code_5m_study_uses_frozen_stage_outputs_and_external_development_holdout() -> None:
    study = json.loads(STUDY.read_text(encoding="utf-8"))

    sources = study["training_sources"]
    assert sources["raw_safe_release_candidates"].endswith("stage0_output\\release_candidates.jsonl")
    assert sources["curated_stage_b_selected"].endswith("stages\\stage_b_selected.jsonl")
    assert study["holdout_contract"]["development"]["stage_b_read"] is False
    assert study["stage_b_isolation"]["utility_available_to_stage_b"] is False
    assert study["benchmark_contamination_boundary"] == "development_comparative_only"


def test_code_5m_execution_is_natural_budget_three_seed_qLoRA() -> None:
    execution = json.loads(EXECUTION.read_text(encoding="utf-8"))

    assert execution["training_arms"] == ["raw_safe_natural", "curated_natural"]
    assert execution["training_recipe"]["development_training_seeds"] == [11, 23, 37]
    assert execution["training_recipe"]["optimizer_steps_by_arm"] == {
        "raw_safe_natural": 429,
        "curated_natural": 156,
    }
    assert execution["benchmark"]["role"] == "development_comparative_only"
    assert execution["stage_b_isolation"]["benchmark_outcomes_available_to_stage_b"] is False


if __name__ == "__main__":
    test_code_5m_study_uses_frozen_stage_outputs_and_external_development_holdout()
    test_code_5m_execution_is_natural_budget_three_seed_qLoRA()
    print("[code-5m-external-validation] frozen inputs and isolated three-seed execution: pass")
