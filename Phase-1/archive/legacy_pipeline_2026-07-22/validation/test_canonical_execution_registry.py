#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "canonical_execution_registry_report.json"
EXPECTED_CANONICAL = [
    "229_build_code_livecodebench_confirmation_summary.py",
    "211_build_code_paper_evidence_report.py",
    "227_build_code_livecodebench_pilot_summary.py",
    "213_build_final_paper_evidence_table.py",
    "190_run_paper_claim_release_gate.py",
    "219_build_domain_composition_audit.py",
    "220_build_coverage_domain_mix_audit.py",
    "221_build_stage_b_policy_contract_audit.py",
    "218_build_paper_claim_consistency_audit.py",
]
FORBIDDEN_FRAGMENTS = ("acquire", "collect", "fetch", "qlora", "train", "training", "generate")


def main() -> int:
    subprocess.run([sys.executable, "222_build_canonical_execution_registry.py"], cwd=PROJECT_DIR, check=True)
    report = load_json(REPORT_PATH)
    canonical_scripts = [entry["script"] for entry in report["canonical_execution_path"]]

    assert report["status"] == "canonical_execution_registry_passed"
    assert canonical_scripts == EXPECTED_CANONICAL
    assert report["scope"] == "paper_evidence_package_rebuild_path"
    assert report["canonical_runner"]["script"] == "run_canonical_paper_evidence.py"
    assert report["canonical_runner"]["source"]["exists"] is True
    assert report["canonical_runner"]["decision_exit_codes"] == [0, 2]
    assert report["active_surface"]["active_entry_points"] == ["00_run_data_eval.py", "run_canonical_paper_evidence.py"]
    assert report["active_surface"]["compatibility_entry_points"] == ["13_run_paper_release.py"]
    assert report["active_surface"]["historical_scripts_are_not_active_entry_points"] is True
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert report["summary"]["canonical_count"] == len(EXPECTED_CANONICAL)
    assert report["summary"]["support_report_count"] >= 6
    assert report["summary"]["historical_numbered_script_count"] > report["summary"]["canonical_count"]
    assert report["canonical_path_is_lightweight_rebuild"] is True
    assert report["missing_expected_outputs"] == []
    assert report["missing_support_reports"] == []
    assert report["forbidden_canonical_script_hits"] == []
    assert "not a full raw-data acquisition" in report["claim_boundary"]

    lowered = " ".join(canonical_scripts).lower()
    assert not any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS)
    print("[canonical-execution-registry] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
