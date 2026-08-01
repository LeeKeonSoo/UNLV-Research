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


def test_target_size_freeze_materializes_binary_only_contract(tmp_path):
    module = _load_script(
        "freeze_redundancy_target_size_development",
        "188_freeze_redundancy_target_size_development.py",
    )
    plan_path = tmp_path / "target_size_plan.json"
    output_dir = tmp_path / "target_size_outputs"
    manifest_path = tmp_path / "blocks_manifest.json"
    manifest = module.build(
        module.DEFAULT_PROXY_PLAN,
        module.DEFAULT_CANONICAL_DECISION,
        plan_path,
        output_dir,
        manifest_path,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "target_size_blocks_materialized"
    assert persisted["blockers"] == []
    assert plan["status"] == "frozen_before_target_size_development_outcomes"
    assert plan["training_arms"] == [
        "binary_current_equal_budget",
        "stageA_random_common_disjoint_equal_budget",
    ]
    assert "log_count_equal_budget" not in plan["training_arms"]
    assert plan["primary_comparison"]["treatment"] == "binary_current_equal_budget"
    assert plan["primary_comparison"]["primary_baseline"] == "stageA_random_common_disjoint_equal_budget"
    assert plan["training_recipe"]["development_training_seeds"] == [11, 23, 37]
    assert plan["utility_scope"].startswith("Stage C validation only")
    assert plan["confirmatory_outcomes_read"] is False
    assert persisted["training_contract"]["exact_tokens_per_arm"] == 327680
    assert persisted["heldout_contract"]["exact_tokens"] == 65536
    for name in [
        "binary_current_equal_budget",
        "stageA_random_common_disjoint_equal_budget",
        "development_code_nll_heldout",
    ]:
        assert Path(persisted["artifacts"][name]["path"]).exists()


def test_qwen3_runner_accepts_safetensors_target_blocks(tmp_path):
    module = _load_script(
        "code_domain_development_runner",
        "141_run_code_domain_development_qlora.py",
    )
    plan = {
        "training_arms": [
            "binary_current_equal_budget",
            "stageA_random_common_disjoint_equal_budget",
        ],
        "training_recipe": {
            "development_training_seeds": [11, 23, 37],
            "optimizer_steps": 40,
        },
    }
    assert module._trained_arms(plan) == [
        "binary_current_equal_budget",
        "stageA_random_common_disjoint_equal_budget",
    ]
    assert module._training_seeds(plan) == [11, 23, 37]
