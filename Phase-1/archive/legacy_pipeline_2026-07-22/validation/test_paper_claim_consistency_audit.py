#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "paper_claim_consistency_audit_report.json"
FINAL_TABLE_PATH = PROJECT_DIR / "outputs" / "validation" / "final_paper_evidence_table.json"
NATURAL_CODE_PATH = PROJECT_DIR / "outputs" / "validation" / "code_domain_natural_budget_stage_c_summary_report.json"


def main() -> int:
    subprocess.run(
        [sys.executable, "229_build_code_livecodebench_confirmation_summary.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "211_build_code_paper_evidence_report.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "213_build_final_paper_evidence_table.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    release_gate = subprocess.run(
        [sys.executable, "190_run_paper_claim_release_gate.py"],
        cwd=PROJECT_DIR,
        check=False,
    )
    assert release_gate.returncode == 0
    subprocess.run(
        [sys.executable, "218_build_paper_claim_consistency_audit.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    report = load_json(REPORT_PATH)
    sections = report["sections"]
    assert report["status"] == "paper_claim_consistency_audit_passed"
    assert report["blockers"] == []
    assert sections["code_domain"]["status"] == "pass"
    assert sections["code_domain"]["current_framework_artifacts_match"] is True
    assert sections["code_domain"]["external_transfer_status"] == (
        "completed_multiseed_external_transfer_inconclusive"
    )
    assert sections["code_domain"]["external_transfer_claim"] == (
        "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation"
    )
    assert sections["code_domain"]["claim_statement"] == "code_domain_curated_natural_budget_improves_nll_and_evalplus_with_fewer_training_tokens"
    assert sections["math_domain"]["status"] == "abstain"
    assert sections["math_domain"]["v3_repairs_v2"] is True
    assert sections["math_domain"]["v3_does_not_beat_raw"] is True
    assert sections["final_evidence_table"]["status"] == "pass"
    final_table = load_json(FINAL_TABLE_PATH)
    natural_code = load_json(NATURAL_CODE_PATH)
    table_rows = {f"{row['domain']}::{row['arm']}": row for row in final_table["rows"]}
    raw_row = table_rows["Code::raw_full_natural"]
    curated_row = table_rows["Code::curated_v2_natural"]
    assert final_table["domain_decisions"]["Code"] == "pass"
    assert raw_row["protocol_id"] == natural_code["schema_version"]
    assert curated_row["protocol_id"] == natural_code["schema_version"]
    assert raw_row["evalplus_macro_pass_rate"] == natural_code["arms"]["raw_full_natural"]["evalplus"]["macro_pass_rate"]
    assert curated_row["evalplus_macro_pass_rate"] == natural_code["arms"]["curated_v2_natural"]["evalplus"]["macro_pass_rate"]
    assert sections["paper_gate"]["status"] == "pass"
    assert sections["paper_gate"]["curation_stage_framework_claim_supported"] is True
    assert sections["paper_gate"]["production_deployment_claim_supported"] is False
    assert "all_domain_improvement_guarantee" in report["forbidden_claims"]
    assert any(
        "code_domain_natural_budget_current_framework_stage_c_summary_report.json" in path
        for path in report["source_sha256"]
    )
    assert "historical_code_positive_requires_current_framework_rerun" not in report["allowed_claims"]
    assert "code_domain_curated_natural_budget_improves_nll_and_evalplus_with_fewer_training_tokens" in report["allowed_claims"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[paper-claim-consistency-audit] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
