#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "validation" / "stage0_release_blocker_report.json"


def test_stage0_release_blocker_report_keeps_production_claim_blocked() -> None:
    subprocess.run([sys.executable, "213_build_record_disposition_audit_report.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "210_build_production_readiness_gate_report.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "212_build_stage0_release_blocker_report.py"], cwd=ROOT, check=True)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["status"] == "stage0_release_blocked_production_guardrails"
    assert report["production_release_allowed"] is False
    assert report["development_evidence_present"] is True
    assert report["production_gate_prototype_status"] == "production_gate_prototype_passed"
    assert report["production_gate_prototype_ready"] is True
    assert report["source_reports"]["production_readiness_gate_report.json"]["status"] == "production_gate_prototype_passed"
    assert set(report["production_blockers"]) == {
        "external_pii_detector_validation_missing",
        "external_secret_detector_validation_missing",
        "license_compliance_validation_missing",
        "benchmark_contamination_validation_missing",
        "adversarial_poisoning_validation_missing",
    }
    assert "not production detector validation" in report["claim_boundary"]


def main() -> int:
    test_stage0_release_blocker_report_keeps_production_claim_blocked()
    print("[stage0-release-blocker] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
