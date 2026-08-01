import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "redundancy_canonical_guardrail_decision",
        ROOT / "187_build_redundancy_canonical_guardrail_decision.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_guardrail_decision_preserves_development_boundary(tmp_path):
    module = _load_script()
    output = tmp_path / "decision.json"
    report = module.build(
        module.DEFAULT_CONTRACT,
        module.DEFAULT_PROXY_DECISION,
        module.DEFAULT_NLL,
        module.DEFAULT_GENERAL_TASK,
        module.DEFAULT_EVALPLUS,
        module.DEFAULT_TARGET_SIZE,
        module.DEFAULT_V2_CONFIRMATORY,
        output,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "canonical_qwen25_0p5b_development_guardrails_passed"
    assert persisted["release_decision"] == "release_supported"
    assert persisted["release_blockers"] == []
    assert persisted["canonical_selector_path"] == "binary_current_equal_budget"
    assert persisted["rejected_candidate_path"] == "log_count_equal_budget"
    assert persisted["confirmatory_outcomes_read"] is True
    assert persisted["target_size_release_decision"] == "release_supported"
    assert persisted["v2_confirmatory_decision"] == "v2_confirmatory_decision_passed"
    assert "general_task_retention" in persisted["evidence"]
    assert "evalplus_development_retention" in persisted["evidence"]
