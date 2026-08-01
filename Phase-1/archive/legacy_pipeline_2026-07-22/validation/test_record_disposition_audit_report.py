#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs" / "validation" / "record_disposition_audit_report.json"


def main() -> int:
    subprocess.run([sys.executable, "213_build_record_disposition_audit_report.py"], cwd=ROOT, check=True)
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "record_disposition_audit_passed"
    assert report["budget_not_selected_is_rejection"] is False
    assert report["retain_all_is_valid"] is True
    assert report["abstain_action_allowed"] is True
    assert set(report["observed_curation_dispositions"]) == {"retained", "rejected", "quarantined"}
    assert set(report["observed_training_budget_dispositions"]) == {
        "not_requested",
        "selected_for_training_budget",
        "budget_not_selected",
    }
    assert report["blockers"] == []

    print("[record-disposition-audit] report contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
