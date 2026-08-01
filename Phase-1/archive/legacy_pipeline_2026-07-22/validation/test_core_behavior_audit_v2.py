#!/usr/bin/env python3
"""Validate the Core behavior audit v2 contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("165_build_core_behavior_audit_v2.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "outputs" / "validation" / "core_construct_validity_review.json",
            ROOT / "outputs" / "validation" / "selector_utility_leakage_audit.json",
            ROOT / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json",
            ROOT / "validation" / "fixtures" / "temporal_code_stage_b_proxy_cases.json",
            ROOT / "outputs" / "validation" / "stage0_hazard_benchmark_report.json",
            ROOT / "outputs" / "validation" / "stage0_detector_validation_report.json",
            ROOT / "outputs" / "validation" / "stage0_detector_heldout_benchmark_report.json",
            ROOT / "outputs" / "validation" / "coverage_domain_fixture_benchmark_report.json",
            ROOT / "outputs" / "validation" / "scoring_schema_separation_audit.json",
            ROOT / "outputs" / "validation" / "real_corpus_stage0_coverage_audit.json",
            tmp_path / "core_behavior_audit_v2.json",
            tmp_path / "core_behavior_audit_v2.md",
        )
    assert report["status"] == "core_behavior_audit_development_checks_passed"
    assert report["metric_validity_status"] == "development_only_not_external_construct_validity"
    assert not report["blockers"]
    assert report["decision"]["release_claim_supported"] is False
    assert report["decision"]["core_metric_validity_fully_proven"] is False
    authority = report["metric_authority"]
    assert authority["reference_quality_score"]["authority"] == "stage_b_selection_signal_only"
    assert authority["shingle_near_duplicate_indicator"]["authority"] == "stage_b_soft_signal_only"
    assert authority["shingle_near_duplicate_indicator"]["hard_gate_supported"] is False
    assert authority["small_lm_probe_gain_score"]["authority"] == "stage_c_validator_only"
    assert "core_behavior_fixture_suite_expanded_but_not_exhaustive" in report["remaining_evidence_gaps"]
    assert "stage0_hazard_fixture_benchmark_exists_but_not_production_detector_validation" not in report["remaining_evidence_gaps"]
    assert "stage0_detector_validation_precheck_not_external_production_benchmark" not in report["remaining_evidence_gaps"]
    assert "stage0_detector_heldout_benchmark_not_external_public_benchmark" in report["remaining_evidence_gaps"]
    assert "coverage_domain_fixture_exists_but_not_real_corpus_metadata_validation" not in report["remaining_evidence_gaps"]
    assert "explicit_domain_metadata_missing_for_true_domain_coverage_claim" in report["remaining_evidence_gaps"]
    assert "real_corpus_stage0_hazard_counts_not_production_detector_validation" in report["remaining_evidence_gaps"]
    assert report["supporting_stage0_hazard_benchmark"]["status"] in {
        "stage0_hazard_fixture_benchmark_passed",
        None,
    }
    assert report["supporting_stage0_detector_validation"]["status"] in {
        "stage0_detector_validation_precheck_passed_with_scope_caveats",
        None,
    }
    assert report["supporting_stage0_detector_heldout_benchmark"]["status"] in {
        "stage0_detector_heldout_benchmark_passed_with_scope_caveats",
        None,
    }
    assert report["supporting_coverage_domain_benchmark"]["status"] in {
        "coverage_domain_fixture_benchmark_passed",
        None,
    }
    assert report["supporting_scoring_schema_separation"]["status"] in {
        "scoring_schema_separation_audit_passed",
        None,
    }
    assert report["supporting_real_corpus_stage0_coverage_audit"]["status"] in {
        "real_corpus_stage0_coverage_audit_passed_with_scope_caveats",
        None,
    }
    by_core = report["core_checks"]
    assert all(row["passed"] for row in by_core["Validity"])
    assert all(row["passed"] for row in by_core["Selection Value Evidence"])
    assert all(row["passed"] for row in by_core["Redundancy"])
    assert all(row["passed"] for row in by_core["Coverage"])
    assert by_core["Utility"], "Utility must have at least one Stage-C boundary check"
    selection_value = by_core["Selection Value Evidence"]
    assert any(row["name"] == "fixture:selection_value_structured_code_beats_trivial_code" for row in selection_value)
    assert any(row["name"] == "fixture:selection_value_concise_guard_beats_empty_test" for row in selection_value)
    assert any(row["name"] == "fixture:selection_value_bug_fix_beats_long_noop_chain" for row in selection_value)
    assert any(row["name"] == "fixture:selection_value_edge_case_test_beats_placeholder_test" for row in selection_value)
    assert any(row["name"] == "retain_all_when_no_training_budget_is_constrained" for row in selection_value)
    assert any(row["name"] == "budget_not_selected_records_remain_in_curated_pool" for row in selection_value)
    assert any(
        row["name"] == "selection_value_evidence_declares_no_hard_reject_authority"
        for row in selection_value
    )
    assert any(row["name"] == "fixture:soft_redundancy_detects_template_saturation" for row in by_core["Redundancy"])
    print("[core-behavior-audit-v2] expanded Core behavior checks pass with explicit evidence gaps: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
