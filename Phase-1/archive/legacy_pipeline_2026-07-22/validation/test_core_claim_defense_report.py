#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "192_build_core_claim_defense_report.py"
    spec = importlib.util.spec_from_file_location("core_claim_defense_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "outputs" / "validation" / "core_behavior_audit_v2.json",
            ROOT / "outputs" / "validation" / "redundancy_validity_benchmark_report.json",
            ROOT / "outputs" / "validation" / "scoring_schema_separation_audit.json",
            ROOT / "outputs" / "validation" / "selector_utility_leakage_audit.json",
            ROOT / "outputs" / "validation" / "paper_claim_release_gate_report.json",
            tmp_path / "core_claim_defense_report.json",
            tmp_path / "core_claim_defense_report.md",
        )
    assert report["status"] == "core_claim_defense_scoped_not_release_ready"
    assert report["claim_decision"]["paper_claim_tier"] == "curation_stage_research_framework"
    assert report["claim_decision"]["curation_stage_framework_claim_supported"] is True
    assert report["claim_decision"]["curation_responsibility_evidence_supported"] is True
    assert report["claim_decision"]["production_deployment_claim_supported"] is False
    assert report["claim_decision"]["intrinsic_quality_claim_supported"] is False
    assert report["claim_decision"]["utility_in_selector_supported"] is False
    assert report["claim_decision"]["current_allowed_surface"] == "curation_stage_research_framework"
    assert report["core_axes"]["Selection Value Evidence"]["allowed_claim"] == "pre_outcome_selection_value_proxy"
    redundancy = report["core_axes"]["Redundancy"]
    assert redundancy["allowed_claim"] == "high_precision_conservative_duplicate_control"
    assert redundancy["current_fixture_recall"] < 1.0
    assert "current_threshold_false_negative_on_labeled_fixture" in redundancy["known_gaps"]
    assert redundancy["evidence_ledger"]["cluster_dropout_decision"] == "hold_challenger"
    assert redundancy["claim_boundary"].startswith("Current canonical threshold is defensible")
    assert "recall_complete_deduplication" in redundancy["not_supported"]
    stage0 = report["core_axes"]["Stage 0 Risk Boundary"]
    assert stage0["heldout_status"] == "stage0_detector_heldout_benchmark_passed_with_scope_caveats"
    assert "production_grade_external_detector_validation" in stage0["not_supported"]
    coverage = report["core_axes"]["Coverage"]
    assert "explicit_domain_metadata_missing_for_true_domain_coverage_claim" in coverage["remaining_scope_gaps"]
    categories = report["claim_decision"]["production_blocker_categories"]
    assert "redundancy_is_high_precision_not_recall_complete" in categories
    assert report["release_gate_blockers"] == []
    assert "production_core_validity_not_supported" in report["production_blockers"]
    print("[core-claim-defense] scoped Core claims and blockers are explicit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
