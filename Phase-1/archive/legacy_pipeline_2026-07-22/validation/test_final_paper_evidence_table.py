#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "outputs" / "validation" / "final_paper_evidence_table.json"


def test_final_evidence_table_contains_current_code_and_math_v3_abstain_rows() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rows = {(row["domain"], row["arm"]): row for row in report["rows"]}

    assert report["status"] == "final_paper_evidence_table_frozen"
    assert report["domain_decisions"]["Code"] == "pass"
    assert report["domain_decisions"]["Math"] == "abstain"
    assert report["external_transfer"]["status"] == "completed_multiseed_external_transfer_inconclusive"
    assert report["external_transfer"]["claim"] == "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation"
    assert set(rows) >= {
        ("Code", "base_no_update"),
        ("Code", "raw_full_natural"),
        ("Code", "curated_v2_natural"),
        ("Math", "base_no_update"),
        ("Math", "raw_full_natural"),
        ("Math", "curated_math_v2_natural"),
        ("Math", "curated_math_v3_natural"),
    }
    assert rows[("Code", "curated_v2_natural")]["decision"] == "pass"
    assert rows[("Math", "curated_math_v2_natural")]["decision"] == "fail"
    assert rows[("Math", "curated_math_v3_natural")]["decision"] == "repair_only_abstain"
    assert rows[("Code", "curated_v2_natural")]["packed_training_tokens"] < rows[("Code", "raw_full_natural")]["packed_training_tokens"]
    assert rows[("Math", "curated_math_v2_natural")]["mean_nll"] > rows[("Math", "raw_full_natural")]["mean_nll"]
    assert rows[("Math", "curated_math_v3_natural")]["mean_nll"] < rows[("Math", "curated_math_v2_natural")]["mean_nll"]
    assert rows[("Math", "curated_math_v3_natural")]["mean_nll"] > rows[("Math", "raw_full_natural")]["mean_nll"]


def main() -> int:
    subprocess.run([sys.executable, "229_build_code_livecodebench_confirmation_summary.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "211_build_code_paper_evidence_report.py"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "213_build_final_paper_evidence_table.py"], cwd=ROOT, check=True)
    test_final_evidence_table_contains_current_code_and_math_v3_abstain_rows()
    print("[final-paper-evidence-table] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
