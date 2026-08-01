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


def test_target_size_development_report_uses_required_guardrail_evidence(tmp_path):
    module = _load_script(
        "build_redundancy_target_size_development_report",
        "189_build_redundancy_target_size_development_report.py",
    )
    output = tmp_path / "target_size_report.json"
    report = module.build(
        module.DEFAULT_PLAN,
        module.DEFAULT_BLOCKS,
        module.DEFAULT_OUTPUT_DIR,
        output,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "target_size_development_passed"
    assert persisted["comparison"]["mean_margin_passed"] is True
    assert persisted["comparison"]["all_seed_direction_positive"] is True
    assert persisted["comparison"]["all_seed_margin_passed"] is False
    assert persisted["guardrail_status"]["release_decision"] == "release_supported"
    assert persisted["guardrail_status"]["missing_guardrails"] == []
    assert persisted["guardrail_status"]["failed_guardrails"] == []
    assert persisted["stage_c_guardrails"]["general_text_retention_nll"]["passed"] is True
    assert persisted["stage_c_guardrails"]["general_task_retention"]["passed"] is True
    assert persisted["stage_c_guardrails"]["evalplus_development"]["passed"] is True
    assert persisted["utility_scope"].startswith("Stage C validation only")
    assert persisted["confirmatory_outcomes_read"] is False
