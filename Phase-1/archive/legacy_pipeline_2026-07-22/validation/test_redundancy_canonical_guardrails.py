import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_guardrail_freeze_excludes_rejected_candidate(tmp_path):
    module = _load_script(
        "freeze_redundancy_canonical_guardrails",
        "186_freeze_redundancy_canonical_guardrails.py",
    )
    output = tmp_path / "guardrails.json"
    report = module.build(
        module.DEFAULT_EXPERIMENT,
        module.DEFAULT_EVALUATION_INPUTS,
        module.DEFAULT_DECISION,
        output,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "frozen_before_canonical_guardrail_outcomes"
    assert persisted["training_arms"] == ["binary_current_equal_budget"]
    assert persisted["training_recipe"]["development_training_seeds"] == [11, 23, 37]
    assert "log_count_equal_budget" in persisted["excluded_arms"]
    assert persisted["required_jobs"]["evalplus_development"]["tasks_per_job"] == 284
    assert persisted["confirmatory_outcomes_read"] is False


def test_guardrail_runners_accept_plan_defined_arm():
    plan = {
        "training_arms": ["binary_current_equal_budget"],
        "training_recipe": {"development_training_seeds": [11, 23, 37]},
    }
    general = _load_script(
        "general_task_guardrail_runner",
        "148_run_code_domain_general_task_guardrail.py",
    )
    evalplus = _load_script(
        "evalplus_sample_runner",
        "143_generate_code_domain_evalplus_samples.py",
    )
    assert general._trained_arms(plan) == ["binary_current_equal_budget"]
    assert general._training_seeds(plan) == [11, 23, 37]
    assert evalplus._trained_arms(plan) == ["binary_current_equal_budget"]
    assert evalplus._training_seeds(plan) == [11, 23, 37]
