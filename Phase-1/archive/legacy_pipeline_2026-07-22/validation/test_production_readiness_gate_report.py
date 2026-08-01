#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "validation" / "test_production_readiness_gate_report.json"
MD_REPORT = ROOT / "outputs" / "validation" / "test_production_readiness_gate_report.md"


def main() -> int:
    subprocess.run(
        [sys.executable, "213_build_record_disposition_audit_report.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "210_build_production_readiness_gate_report.py",
            "--output",
            str(REPORT),
            "--md-output",
            str(MD_REPORT),
        ],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "production_gate_prototype_passed"
    assert report["readiness_level"] == "R1_production_gate_prototype"
    assert report["paper_claim_ready"] is True
    assert report["production_certified"] is False
    assert report["stage0_hazard_fixture_passed"] is True
    assert report["utility_leakage_audit_passed"] is True
    assert report["record_disposition_audit_passed"] is True
    assert "external_detector_validation_missing" in report["r3_blockers"]
    assert "production_certification" in report["forbidden_claims"]

    print("[production-readiness-gate] report contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
