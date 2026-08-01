#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "coverage_domain_mix_audit_report.json"


def main() -> int:
    subprocess.run([sys.executable, "219_build_domain_composition_audit.py"], cwd=PROJECT_DIR, check=True)
    subprocess.run([sys.executable, "220_build_coverage_domain_mix_audit.py"], cwd=PROJECT_DIR, check=True)
    report = load_json(REPORT_PATH)

    assert report["status"] == "coverage_domain_mix_audit_passed_with_scope_boundary"
    assert report["coverage_role"] == "composition_and_collapse_audit_not_utility_evidence"
    assert report["input_statuses"]["coverage_fixture_report"] == "coverage_domain_fixture_benchmark_passed"
    assert report["input_statuses"]["domain_composition_report"] == "domain_composition_audit_completed"
    assert report["input_statuses"]["domain_mix_contract"] == "domain_mix_contract_frozen"
    assert report["target_mix"]["status"] == "not_declared_for_current_paper_evidence"
    assert report["target_mix"]["target_mix_claim_allowed"] is False
    assert report["target_mix"]["observed_composition_claim_allowed"] is True
    assert report["coverage_scope"]["true_domain_claim_policy"] == "requires_explicit_metadata_or_declared_contract"
    assert report["coverage_scope"]["current_scope"] == "observed_paper_domain_arm_composition"
    assert report["domain_share_drift"]["Code"] < 0.0
    assert report["domain_share_drift"]["Math"] > 0.0
    assert report["max_abs_domain_share_drift"] > 0.19
    assert "coverage_proves_utility" in report["forbidden_claims"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    print("[coverage-domain-mix-audit] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
