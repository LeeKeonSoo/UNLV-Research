#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "canonical_paper_evidence_run_report.json"


def main() -> int:
    completed = subprocess.run(
        [sys.executable, "run_canonical_paper_evidence.py"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Canonical paper-evidence rebuild:" in completed.stdout
    assert "229_build_code_livecodebench_confirmation_summary.py" in completed.stdout
    assert "211_build_code_paper_evidence_report.py" in completed.stdout
    assert "218_build_paper_claim_consistency_audit.py" in completed.stdout
    assert "Decision exit code 2" in completed.stdout
    executed = subprocess.run(
        [sys.executable, "run_canonical_paper_evidence.py", "--execute"],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "canonical_paper_evidence_run_passed"
    assert len(report["results"]) == 9
    assert report["results"][0]["script"] == "229_build_code_livecodebench_confirmation_summary.py"
    assert report["results"][4]["script"] == "190_run_paper_claim_release_gate.py"
    assert report["results"][4]["decision_blocked"] is False
    assert all(item["succeeded"] is True for item in report["results"])
    print("[canonical-paper-evidence-runner] plan: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
