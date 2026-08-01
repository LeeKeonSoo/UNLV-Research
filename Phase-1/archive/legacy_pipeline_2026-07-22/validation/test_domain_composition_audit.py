#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "domain_composition_audit_report.json"


def main() -> int:
    subprocess.run(
        [sys.executable, "219_build_domain_composition_audit.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    report = load_json(REPORT_PATH)
    rows = {row["domain"]: row for row in report["domain_rows"]}

    assert report["status"] == "domain_composition_audit_completed"
    assert report["contract_mode"] == "observed_paper_domain_arm_composition"
    assert report["target_domain_mix_status"] == "not_declared_for_current_paper_evidence"
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert "joint_production_mix_certification" in report["forbidden_claims"]

    assert set(rows) == {"Code", "Math"}
    assert rows["Code"]["raw"]["packed_training_tokens"] == 980992
    assert rows["Code"]["curated"]["packed_training_tokens"] == 385024
    assert rows["Code"]["packed_token_reduction_fraction"] > 0.60
    assert rows["Code"]["curated_decision"] == "pass"
    assert report["decision_contract_pass"] is True

    assert rows["Math"]["raw"]["packed_training_tokens"] == 1120256
    assert rows["Math"]["curated"]["packed_training_tokens"] == 1026048
    assert rows["Math"]["packed_token_reduction_fraction"] < 0.10
    assert rows["Math"]["curated_decision"] == "repair_only_abstain"

    raw_mix = report["mixes"]["raw"]["shares"]
    curated_mix = report["mixes"]["curated"]["shares"]
    assert abs(sum(raw_mix.values()) - 1.0) < 0.000001
    assert abs(sum(curated_mix.values()) - 1.0) < 0.000001
    assert curated_mix["Code"] < raw_mix["Code"]
    assert curated_mix["Math"] > raw_mix["Math"]
    print("[domain-composition-audit] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
