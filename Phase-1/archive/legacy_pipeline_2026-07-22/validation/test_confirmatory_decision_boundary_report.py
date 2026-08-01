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
    path = ROOT / "195_build_confirmatory_decision_boundary_report.py"
    spec = importlib.util.spec_from_file_location("confirmatory_decision_boundary_report", path)
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
            ROOT / "outputs" / "validation" / "stage_c_training_validation_report.json",
            ROOT / "outputs" / "validation" / "paper_claim_release_gate_report.json",
            tmp_path / "confirmatory_decision_boundary_report.json",
            tmp_path / "confirmatory_decision_boundary_report.md",
        )
    assert report["status"] == "confirmatory_decision_curation_stage_claim_passed"
    assert report["final_decision"] == "curation_stage_claim_pass"
    assert report["claim_decision"]["target_nll_confirmatory_effect_supported"] is True
    assert report["claim_decision"]["required_confirmatory_guardrails_complete"] is True
    assert report["claim_decision"]["curation_stage_claim_supported"] is True
    assert report["claim_decision"]["production_deployment_claim_supported"] is False
    assert report["claim_decision"]["stage_b_tuning_allowed"] is False
    assert report["nll_evidence"]["nll_gate_status"] == "passed"
    assert report["guardrail_decision"]["incomplete_guardrails"] == []
    assert report["guardrail_decision"]["passed_guardrails"] == [
        "evalplus_confirmatory",
        "general_task_retention",
        "general_text_nll_retention",
    ]
    assert report["release_gate_blockers"] == []
    assert "production_core_validity_not_supported" in report["production_blockers"]
    assert "complete_production_core_validity_before_deployment_claims" in report["remaining_actions"]
    print("[confirmatory-decision-boundary] complete guardrails plus paper gate yields curation-stage claim pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
