from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    subprocess.run(
        [sys.executable, "229_build_code_livecodebench_confirmation_summary.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [sys.executable, "211_build_code_paper_evidence_report.py"],
        cwd=ROOT,
        check=True,
    )
    module = _load("190_run_paper_claim_release_gate.py")
    with tempfile.TemporaryDirectory() as tmp:
        report = module.build(
            ROOT / "outputs" / "validation" / "core_behavior_audit_v2.json",
            ROOT / "outputs" / "validation" / "selector_utility_leakage_audit.json",
            ROOT / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json",
            ROOT / "validation" / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json",
            ROOT / "validation" / "frozen_contracts" / "redundancy_target_size_qwen3_4b_development_report.json",
            ROOT / "outputs" / "validation" / "code_paper_evidence_report.json",
            Path(tmp) / "paper_claim_release_gate_report.json",
        )
    assert report["status"] == "paper_curation_stage_claim_gate_passed"
    assert report["supported"] is True
    assert report["paper_claim_tier"] == "curation_stage_research_framework"
    assert report["curation_stage_framework_claim_supported"] is True
    assert report["production_deployment_claim_supported"] is False
    assert report["blockers"] == []
    assert "production_core_validity_not_supported" in report["production_blockers"]
    assert "canonical_guardrail_release_not_supported:abstain_not_a_production_release" not in report["blockers"]
    assert not any(str(row).startswith("v2_confirmatory_") for row in report["blockers"])
    assert "target_size_missing_required_guardrails" not in report["blockers"]
    assert "target_size_failed_required_guardrails" not in report["blockers"]
    assert not any(str(row).startswith("target_size_") for row in report["blockers"])
    print("[paper-claim-release-gate] current-framework curation-stage claim: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
