#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "194_build_stage_c_training_validation_report.py"
    spec = importlib.util.spec_from_file_location("stage_c_training_validation_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json",
            ROOT / "outputs" / "validation" / "stage_c_guardrail_gap_report.json",
            ROOT / "validation" / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json",
            ROOT / "validation" / "frozen_contracts" / "redundancy_target_size_qwen3_4b_development_report.json",
            tmp_path / "stage_c_training_validation_report.json",
            tmp_path / "stage_c_training_validation_report.md",
        )
    assert report["status"] == "stage_c_training_validation_nll_supported_curation_claim_ready"
    assert report["claim_decision"]["target_nll_training_effect_supported"] is True
    assert report["claim_decision"]["curation_stage_paper_claim_supported"] is True
    assert report["claim_decision"]["production_deployment_claim_supported"] is False
    assert report["claim_decision"]["confirmatory_complete"] is True
    assert report["claim_decision"]["utility_in_selector_supported"] is False
    assert report["claim_decision"]["target_size_guardrails_closed"] is True
    assert report["v2_confirmatory_training"]["nll_gate_status"] == "passed"
    assert report["v2_confirmatory_training"]["curated_vs_stageA_random_mean_nll_reduction"] > 0.003
    assert report["guardrail_gap"]["status"] == "stage_c_guardrail_gaps_closed"
    assert report["guardrail_gap"]["incomplete_guardrails"] == []
    assert report["target_size_training"]["status"] == "target_size_development_passed"
    assert report["target_size_training"]["missing_guardrails"] == []
    assert report["target_size_training"]["release_decision"] == "release_supported"
    assert report["canonical_proxy_training"]["status"] == "canonical_qwen25_0p5b_development_guardrails_passed"
    assert report["canonical_proxy_training"]["release_decision"] == "release_supported"
    assert "canonical_proxy_guardrail_release_decision_abstains" not in report["remaining_evidence_gaps"]
    assert "production_deployment_core_validity_gap" in report["remaining_evidence_gaps"]
    assert "target_size_general_text_general_task_evalplus_guardrails_missing" not in report["remaining_evidence_gaps"]
    print("[stage-c-training-validation] NLL effect supports curation-stage claim while production stays blocked: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
