#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "stage_b_policy_contract_audit_report.json"


def main() -> int:
    subprocess.run([sys.executable, "213_build_record_disposition_audit_report.py"], cwd=PROJECT_DIR, check=True)
    subprocess.run([sys.executable, "220_build_coverage_domain_mix_audit.py"], cwd=PROJECT_DIR, check=True)
    subprocess.run([sys.executable, "221_build_stage_b_policy_contract_audit.py"], cwd=PROJECT_DIR, check=True)
    report = load_json(REPORT_PATH)

    assert report["status"] == "stage_b_policy_contract_audit_passed"
    assert report["stage_b_role"] == "optional_budget_allocation_over_stage_a_survivors"
    assert report["activation"]["no_binding_budget_action"] == "retain_all"
    assert report["activation"]["binding_budget_required"] is True
    assert report["disposition_semantics"]["retain_all_is_valid"] is True
    assert report["disposition_semantics"]["budget_not_selected_is_rejection"] is False
    assert report["disposition_semantics"]["budget_not_selected_is_low_quality"] is False
    assert report["selector_boundary"]["utility_leakage_status"] == "selector_utility_leakage_audit_passed"
    assert report["selector_boundary"]["unexpected_stage_b_evidence_keys"] == []
    assert report["coverage_boundary"]["observed_composition_claim_allowed"] is True
    assert report["coverage_boundary"]["target_mix_claim_allowed"] is False
    assert "curation_requires_dataset_shrinkage" in report["forbidden_claims"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[stage-b-policy-contract-audit] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
