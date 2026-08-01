#!/usr/bin/env python3
"""Validation helpers for the generic data evaluation pipeline."""

from __future__ import annotations

import json
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from data_eval_common import (
    ALL_METRICS,
    CORE_SUBSET_METRICS,
    CORE_SELECTION_METRICS,
    DASHBOARD_PATH,
    DIAGNOSTIC_METRICS,
    METRIC_SPEC_PATH,
    METRIC_SPEC_SCHEMA_VERSION,
    OUTPUT_DIR,
    RUN_MANIFEST_PATH,
    RUN_SUMMARY_PATH,
    SCHEMA_VERSION,
    SCORED_DIR,
    SUBSETS_DIR,
    UTILITY_PROBE_RESULTS_PATH,
    count_nonempty_lines_resilient,
    fingerprint_files,
    iter_jsonl_records_resilient,
    iter_nonempty_lines_resilient,
    load_json,
    save_json,
    scoring_metric_spec_fingerprint,
)
from reports.dashboard import build_dashboard
from reports.metric_maturity import build_metric_maturity_snapshot


SCORING_MANIFEST_PATH = SCORED_DIR / "scoring_manifest.json"
VALIDATION_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "full_validation_report.json"
PROPERTY_BENCHMARK_DIR = Path(__file__).resolve().parent / "outputs" / "validation" / "property_benchmarks"
UTILITY_SENSITIVITY_AUDIT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "utility_sensitivity_audit.json"
UTILITY_POWER_SWEEP_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "utility_probe_power_sweep.json"
CURATION_READINESS_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "curation_readiness_report.json"
CURATION_READINESS_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "curation_readiness_report.md"
STAGE_C_PROTOCOL_DECISION_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "stage_c_protocol_decision_report.json"
STAGE_C_PROTOCOL_DECISION_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "stage_c_protocol_decision_report.md"
STRICT_BASELINE_CONTROL_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "strict_baseline_control_report.json"
STRICT_BASELINE_CONTROL_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "strict_baseline_control_report.md"
CURATION_DECISION_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "curation_decision_report.json"
CURATION_DECISION_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "curation_decision_report.md"
PAPER_EVIDENCE_TABLE_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "paper_evidence_table.json"
PAPER_EVIDENCE_TABLE_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "paper_evidence_table.md"
PAPER_EVIDENCE_TABLE_CSV_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "paper_evidence_table.csv"
STAGE0_CONTRACT_VALIDATION_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "stage0_contract_validation.json"
STAGE0_PROCESSING_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "stage0" / "stage0_processing_report.json"
OPENWEBTEXT2_SLICE_DIAGNOSTIC_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "openwebtext2_slice_diagnostic.json"
SELECTOR_BASELINE_AUDIT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "selector_baseline_audit.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "utility_transfer_gap_report.json"
UTILITY_TRANSFER_GAP_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "utility_transfer_gap_report.md"
CORE_PROXY_ALIGNMENT_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "core_proxy_alignment_report.json"
CORE_PROXY_ALIGNMENT_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "core_proxy_alignment_report.md"
CORE_PROXY_CALIBRATION_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "core_proxy_calibration_report.json"
CORE_PROXY_CALIBRATION_REPORT_MD_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "core_proxy_calibration_report.md"
ANTI_MEMORIZATION_PROBE_REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "anti_memorization_probe_report.json"
POLICY_ABLATION_AUDIT_PATH = Path(__file__).resolve().parent / "outputs" / "validation" / "policy_ablation_audit.json"
ANTI_MEMORIZATION_PROBE_REPORT_GLOB = "anti_memorization_probe_report*.json"
UTILITY_BASELINE_COMPARISON_GLOB = "utility_baseline_comparison_*.json"
UTILITY_MATCHING_DECOMPOSITION_GLOB = "utility_matching_decomposition_*.json"
CANDIDATE_PROFILE_COMPARISON_GLOB = "candidate_profile_comparison*.json"
SLM_UPDATE_EXPERIMENT_MANIFEST_GLOB = "slm_update_experiments/*/manifest.json"
SLM_UPDATE_FROZEN_PLAN_NAME = "frozen_training_plan.json"
TEMPORAL_CODE_STAGE_B_REPORT_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "stage_b_smoke_report.json"
TEMPORAL_CODE_STAGE_B_SELECTED_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "train_selected.jsonl"
TEMPORAL_CODE_STAGE_B_BASELINE_PATH = OUTPUT_DIR / "temporal_code_collection" / "stage_b_smoke" / "train_stage_a_random_disjoint.jsonl"
TEMPORAL_CODE_STAGE_B_PROXY_VALIDATION_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_proxy_validation.json"
TEMPORAL_CODE_STAGE_B_INDEX_EQUIVALENCE_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_index_equivalence.json"
TEMPORAL_CODE_STAGE_B_BLIND_PACKET_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_packet.json"
TEMPORAL_CODE_STAGE_B_BLIND_KEY_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review" / "blind_review_key.json"
TEMPORAL_CODE_STAGE_B_BLIND_ANALYSIS_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_blind_review_analysis.json"
TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_DIR = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review"
TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_ANALYSIS_PATH = OUTPUT_DIR / "validation" / "temporal_code_stage_b_multi_review_analysis.json"
TEMPORAL_CODE_PROXY_REVIEW_EXPANSION_PATH = OUTPUT_DIR / "temporal_code_collection" / "proxy_review_expansion_4" / "proxy_review_expansion_report.json"
METRIC_EVIDENCE_AUDIT_PATH = Path(__file__).resolve().parent / "configs" / "metric_evidence_audit.json"
TEMPORAL_CODE_STAGE_B_ABLATION_PROTOCOL_PATH = Path(__file__).resolve().parent / "configs" / "temporal_code_stage_b_ablation_protocol_v1.json"
TEMPORAL_CODE_BROAD_FREEZE_CONTRACT_PATH = Path(__file__).resolve().parent / "configs" / "temporal_code_broad_collection_freeze_v1.json"
TEMPORAL_CODE_BROAD_MANIFEST_PATH = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_repository_manifest.json"
TEMPORAL_CODE_BROAD_TRANCHE_CONTRACT_PATH = Path(__file__).resolve().parent / "configs" / "temporal_code_broad_tranche_v1.json"
TEMPORAL_CODE_BROAD_TRANCHE_PLAN_PATH = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_tranche_plan.json"
TEMPORAL_CODE_BROAD_TEST_COMMANDS_PATH = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_broad_test_commands_v1.json"
TEMPORAL_CODE_BROAD_READINESS_PATH = OUTPUT_DIR / "validation" / "temporal_code_broad_tranche_readiness.json"
TEMPORAL_CODE_BROAD_ABLATIONS_PATH = OUTPUT_DIR / "validation" / "temporal_code_broad_stage_b_ablations.json"
TEMPORAL_CODE_PATH_STRATIFIED_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "configs" / "temporal_code_path_stratified_tranche_v2.json"
)
TEMPORAL_CODE_PATH_STRATIFIED_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_path_stratified_tranche_plan_v2.json"
)
TEMPORAL_CODE_PATH_STRATIFIED_READINESS_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_path_stratified_tranche_readiness.json"
)
TEMPORAL_CODE_PATH_STRATIFIED_ABLATIONS_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_path_stratified_stage_b_ablations.json"
)
TEMPORAL_CODE_CONFIRMATORY_EXPANSION_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_confirmatory_execution_expansion_plan.json"
)
TEMPORAL_CODE_DEVELOPMENT_EXPANSION_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_development_execution_expansion_plan.json"
)
TEMPORAL_CODE_DEVELOPMENT_EXPANSION_READINESS_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_development_expansion_readiness.json"
)
TEMPORAL_CODE_NATIVE_EXECUTION_COMMANDS_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_development_native_test_commands_v1.json"
)
TEMPORAL_CODE_NATIVE_EXECUTION_REFINEMENT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_native_execution_refinement_report.json"
)
TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_development_fresh_expansion_plan.json"
)
TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_development_fresh_expansion_report.json"
)
TEMPORAL_CODE_EXECUTION_SUPPORT_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_execution_support_report.json"
)
TEMPORAL_CODE_EXECUTABLE_TASK_HARNESS_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_executable_task_harness_plan.json"
)
TEMPORAL_CODE_SWEBENCH_HARNESS_METADATA_PROFILE_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "swebench_harness_metadata_profile.json"
)
TEMPORAL_CODE_EVALPLUS_GUARDRAIL_PREVALIDATION_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
)
TEMPORAL_CODE_EVALPLUS_GUARDRAIL_SPLIT_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "evalplus_guardrail_split_plan.json"
)
TEMPORAL_CODE_RETENTION_GUARDRAIL_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retention_guardrail_plan.json"
)
TEMPORAL_CODE_PRIMARY_SOURCE_ASSESSMENT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_primary_executable_source_assessment.json"
)
TEMPORAL_CODE_FORWARD_E2_PILOT_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_forward_e2_pilot_report.json"
)
TEMPORAL_CODE_FORWARD_E2_PRODUCTIVITY_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_forward_e2_productivity_report.json"
)
TEMPORAL_CODE_FORWARD_DEVELOPMENT_SNAPSHOT_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_forward_development_snapshot_report.json"
)
TEMPORAL_CODE_FORWARD_DEVELOPMENT_ACCUMULATION_PLAN_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_development_accumulation_plan.json"
)
TEMPORAL_CODE_FORWARD_DISCOVERY_CAPACITY_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_forward_discovery_capacity_report.json"
)
TEMPORAL_CODE_FORWARD_COLLECTION_SCHEDULE_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_collection_schedule.json"
)
TEMPORAL_CODE_FORWARD_CANDIDATE_LEDGER_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_forward_candidate_ledger.json"
)
TEMPORAL_CODE_FORWARD_OPERATIONS_STATUS_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_forward_operations_status.json"
)
TEMPORAL_CODE_RETROSPECTIVE_DEVELOPMENT_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_retrospective_development_report.json"
)
TEMPORAL_CODE_RETROSPECTIVE_EXPANSION_SCHEDULE_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retrospective_expansion_schedule.json"
)
TEMPORAL_CODE_RETROSPECTIVE_COMBINED_LEDGER_PATH = (
    OUTPUT_DIR / "temporal_code_collection" / "temporal_code_retrospective_combined_candidate_ledger.json"
)
TEMPORAL_CODE_RETROSPECTIVE_OPERATIONS_STATUS_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_retrospective_operations_status.json"
)
TEMPORAL_CODE_RETROSPECTIVE_E2_CAPACITY_AUDIT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_retrospective_e2_capacity_audit.json"
)
TEMPORAL_CODE_STAGE_C_SMOKE_CONTRACT_PATH = (
    Path(__file__).resolve().parent / "configs" / "temporal_code_stage_c_smoke_qwen3_4b_v1.json"
)
TEMPORAL_CODE_STAGE_C_SMOKE_ARM_MANIFEST_PATH = (
    OUTPUT_DIR / "temporal_code_stage_c_smoke_qwen3_4b_v1" / "frozen_smoke_arm_manifest.json"
)
TEMPORAL_CODE_STAGE_C_SMOKE_BLOCK_MANIFEST_PATH = (
    OUTPUT_DIR / "temporal_code_stage_c_smoke_qwen3_4b_v1" / "token_blocks" / "block_manifest.json"
)
TEMPORAL_CODE_STAGE_C_SMOKE_REPORT_PATH = (
    OUTPUT_DIR / "validation" / "temporal_code_stage_c_smoke_report.json"
)
REPLICATE_PRESET_RE = re.compile(r"^(?P<family>.+)_b(?P<replicate>\d+)$")
LEGACY_VARIANT_PROFILE_ORDER = ("strict", "balanced", "coverage_preserving")
METRIC_ROLES = {"gate", "selection_signal", "subset_validator", "diagnostic"}
METRIC_STATUSES = {"paper_backed", "paper_aligned", "diagnostic", "deprecated_diagnostic"}
VALIDATION_SCOPES = frozenset({"canonical", "full"})
ORTHOGONALITY_MAX_ABS_SPEARMAN = 0.92
ORTHOGONALITY_SAMPLE_LIMIT = 30000
THEORY_AXIS_EXPECTED = {
    "structural_validity_gate": "Validity",
    "structural_validity_score": "Validity",
    "reference_quality_score": "Selection Value Evidence",
    "exact_duplicate_indicator": "Redundancy",
    "shingle_near_duplicate_indicator": "Redundancy",
    "shingle_near_duplicate_risk_score": "Redundancy",
    "subset_coverage_retention_score": "Coverage",
    "small_lm_probe_gain_score": "Utility",
}
SLM_UPDATE_REQUIRED_ARMS = {
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
    "stageA_all_reference",
    "raw_all_reference",
}
SLM_UPDATE_REQUIRED_TRAINING_RUNS = SLM_UPDATE_REQUIRED_ARMS | {"base_no_update"}
SLM_UPDATE_REQUIRED_EVALUATION = {
    "held_out_new_data_distribution",
    "general_capability_benchmarks",
    "forgetting_or_regression_suite",
    "benchmark_contamination_audit",
    "domain_source_slice_analysis",
    "training_stability_and_seed_variance",
    "cost_and_retained_token_efficiency",
}
SLM_UPDATE_EQUAL_BUDGET_ARMS = (
    "curated_equal_budget",
    "stageA_random_equal_budget",
    "raw_random_equal_budget",
)


@dataclass
class ValidationItem:
    name: str
    ok: bool
    details: Dict[str, Any]


def _count_lines(path: Path) -> int:
    return count_nonempty_lines_resilient(path)


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _framework_status(payload: Dict[str, Any]) -> str | None:
    implication = payload.get("framework_implication") if isinstance(payload, dict) else None
    if isinstance(implication, dict):
        status = implication.get("status")
        return str(status) if status is not None else None
    if isinstance(implication, str):
        return implication
    return None


def includes_historical_evidence(scope: str) -> bool:
    if scope not in VALIDATION_SCOPES:
        raise ValueError(f"unsupported validation scope: {scope}")
    return scope == "full"


def _validate_temporal_code_stage_b(items: List[ValidationItem]) -> None:
    if not TEMPORAL_CODE_STAGE_B_REPORT_PATH.exists():
        items.append(
            ValidationItem(
                name="temporal_code_stage_b_smoke_present",
                ok=False,
                details={"path": str(TEMPORAL_CODE_STAGE_B_REPORT_PATH), "reason": "report missing"},
            )
        )
        return
    report = load_json(TEMPORAL_CODE_STAGE_B_REPORT_PATH)
    contract = report.get("stage_b_contract") if isinstance(report.get("stage_b_contract"), dict) else {}
    objective = contract.get("objective") if isinstance(contract.get("objective"), dict) else {}
    forbidden = set(_as_str_list(objective.get("forbidden_signals")))
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    abstained = report.get("operational_decision") == "insufficient_usable_data"
    isolation = report.get("isolation") if isinstance(report.get("isolation"), dict) else {}
    coverage = report.get("coverage_support") if isinstance(report.get("coverage_support"), dict) else {}
    selected = list(iter_jsonl_records_resilient(TEMPORAL_CODE_STAGE_B_SELECTED_PATH))
    baseline = list(iter_jsonl_records_resilient(TEMPORAL_CODE_STAGE_B_BASELINE_PATH))
    selected_ids = {str(row.get("chunk_uid")) for row in selected}
    baseline_ids = {str(row.get("chunk_uid")) for row in baseline}

    items.append(
        ValidationItem(
            name="temporal_code_stage_b_contract",
            ok=report.get("schema_version") == "temporal-code-stage-b-smoke-report-v1"
            and contract.get("input") == "train split Stage-A-pass chunks only"
            and {"Utility", "benchmark outcomes", "development outcomes", "confirmatory outcomes"}.issubset(forbidden)
            and objective.get("redundancy_search_mode") == "indexed_exact"
            and report.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_B_REPORT_PATH), "forbidden_signals": sorted(forbidden)},
        )
    )
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_train_only_isolation",
            ok=isolation.get("selection_input_split") == "train"
            and isolation.get("development_read") is False
            and isolation.get("confirmatory_read") is False
            and all(row.get("split") == "train" and row.get("stage_a_pass") is True for row in [*selected, *baseline]),
            details={"isolation": isolation},
        )
    )
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_disjoint_equal_budget_baseline",
            ok=summary.get("selected_and_baseline_disjoint") is True
            and not selected_ids.intersection(baseline_ids)
            and (
                float(summary.get("baseline_to_selected_token_ratio") or 0.0) >= 0.99
                or (
                    abstained
                    and int(summary.get("input_train_stage_a_pass_chunks") or 0) == 0
                    and not selected
                    and not baseline
                )
            ),
            details={
                "selected_count": len(selected),
                "baseline_count": len(baseline),
                "overlap_count": len(selected_ids.intersection(baseline_ids)),
                "baseline_to_selected_token_ratio": summary.get("baseline_to_selected_token_ratio"),
            },
        )
    )
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_coverage_support",
            ok=coverage.get("all_observed_values_retained") is True
            and coverage.get("all_distribution_floors_passed") is True
            and all(not values for values in (coverage.get("selected_missing_observed_values") or {}).values()),
            details={
                "selected_missing_observed_values": coverage.get("selected_missing_observed_values"),
                "all_distribution_floors_passed": coverage.get("all_distribution_floors_passed"),
            },
        )
    )
    ablation = load_json(TEMPORAL_CODE_STAGE_B_ABLATION_PROTOCOL_PATH) if TEMPORAL_CODE_STAGE_B_ABLATION_PROTOCOL_PATH.exists() else {}
    ablation_arms = ablation.get("arms") if isinstance(ablation.get("arms"), dict) else {}
    ablation_forbidden = set(_as_str_list((ablation.get("shared_contract") or {}).get("forbidden_selector_signals")))
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_frozen_ablation_protocol",
            ok=ablation.get("schema_version") == "temporal-code-stage-b-ablation-protocol-v1"
            and ablation.get("status") == "frozen_before_target_model_development_results"
            and {
                "full_selector",
                "quality_only",
                "redundancy_only",
                "no_coverage_support",
                "stageA_random_equal_token",
                "raw_random_equal_token",
            }.issubset(ablation_arms)
            and {"human or LLM review labels", "Utility", "benchmark outcomes", "development outcomes", "confirmatory outcomes"}.issubset(ablation_forbidden)
            and (ablation.get("confirmatory_rule") or {}).get("policy_weights_thresholds_and_coverage_constraints_frozen") is True,
            details={"path": str(TEMPORAL_CODE_STAGE_B_ABLATION_PROTOCOL_PATH), "arms": sorted(ablation_arms)},
        )
    )
    broad_freeze = load_json(TEMPORAL_CODE_BROAD_FREEZE_CONTRACT_PATH) if TEMPORAL_CODE_BROAD_FREEZE_CONTRACT_PATH.exists() else {}
    items.append(
        ValidationItem(
            name="temporal_code_broad_freeze_contract",
            ok=broad_freeze.get("schema_version") == "temporal-code-broad-collection-freeze-v1"
            and broad_freeze.get("status") == "frozen_before_broad_content_fetch"
            and (broad_freeze.get("content_fetch_limits") or {}).get("issue_and_pull_request_prose")
            == "do_not_fetch_for_training_payload"
            and broad_freeze.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_BROAD_FREEZE_CONTRACT_PATH)},
        )
    )
    broad_manifest = load_json(TEMPORAL_CODE_BROAD_MANIFEST_PATH) if TEMPORAL_CODE_BROAD_MANIFEST_PATH.exists() else {}
    broad_repositories = (
        broad_manifest.get("repositories") if isinstance(broad_manifest.get("repositories"), dict) else {}
    )
    broad_summary = broad_manifest.get("summary") if isinstance(broad_manifest.get("summary"), dict) else {}
    items.append(
        ValidationItem(
            name="temporal_code_broad_frozen_manifest",
            ok=broad_manifest.get("schema_version") == "temporal-code-broad-repository-manifest-v1"
            and broad_manifest.get("status") == "frozen_before_broad_content_fetch"
            and int(broad_summary.get("frozen_repository_count") or 0) == len(broad_repositories)
            and all(int(value or 0) > 0 for value in (broad_summary.get("split_counts") or {}).values())
            and all(row.get("membership_is_training_approval") is False for row in broad_repositories.values())
            and broad_manifest.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_BROAD_MANIFEST_PATH), "summary": broad_summary},
        )
    )
    tranche_contract = load_json(TEMPORAL_CODE_BROAD_TRANCHE_CONTRACT_PATH) if TEMPORAL_CODE_BROAD_TRANCHE_CONTRACT_PATH.exists() else {}
    tranche_forbidden = set(_as_str_list(tranche_contract.get("selection_forbids")))
    items.append(
        ValidationItem(
            name="temporal_code_broad_tranche_contract",
            ok=tranche_contract.get("schema_version") == "temporal-code-broad-tranche-v1"
            and tranche_contract.get("status") == "frozen_before_tranche_content_fetch"
            and {"Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(tranche_forbidden),
            details={"path": str(TEMPORAL_CODE_BROAD_TRANCHE_CONTRACT_PATH)},
        )
    )
    tranche_plan = load_json(TEMPORAL_CODE_BROAD_TRANCHE_PLAN_PATH) if TEMPORAL_CODE_BROAD_TRANCHE_PLAN_PATH.exists() else {}
    tranche_summary = tranche_plan.get("summary") if isinstance(tranche_plan.get("summary"), dict) else {}
    items.append(
        ValidationItem(
            name="temporal_code_broad_tranche_plan",
            ok=tranche_plan.get("schema_version") == "temporal-code-broad-tranche-plan-v1"
            and tranche_plan.get("status") == "frozen_before_tranche_content_fetch"
            and int(tranche_summary.get("repository_count") or 0) == 20
            and tranche_summary.get("split_counts") == {"train": 12, "development": 4, "confirmatory": 4}
            and tranche_plan.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_BROAD_TRANCHE_PLAN_PATH), "summary": tranche_summary},
        )
    )
    broad_commands = load_json(TEMPORAL_CODE_BROAD_TEST_COMMANDS_PATH) if TEMPORAL_CODE_BROAD_TEST_COMMANDS_PATH.exists() else {}
    command_forbidden = set(_as_str_list(broad_commands.get("forbidden_inputs")))
    items.append(
        ValidationItem(
            name="temporal_code_broad_test_commands_frozen",
            ok=broad_commands.get("schema_version") == "temporal-code-broad-test-commands-v1"
            and broad_commands.get("status") == "frozen_before_execution"
            and int((broad_commands.get("summary") or {}).get("repository_count") or 0) == 20
            and (broad_commands.get("isolation_contract") or {}).get("host_execution_forbidden") is True
            and {"Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(command_forbidden),
            details={"path": str(TEMPORAL_CODE_BROAD_TEST_COMMANDS_PATH), "summary": broad_commands.get("summary")},
        )
    )
    broad_readiness = load_json(TEMPORAL_CODE_BROAD_READINESS_PATH) if TEMPORAL_CODE_BROAD_READINESS_PATH.exists() else {}
    readiness_contracts = (
        broad_readiness.get("engineering_contracts")
        if isinstance(broad_readiness.get("engineering_contracts"), dict)
        else {}
    )
    items.append(
        ValidationItem(
            name="temporal_code_broad_tranche_readiness_boundary",
            ok=broad_readiness.get("schema_version") == "temporal-code-broad-tranche-readiness-v1"
            and broad_readiness.get("status") == "stage_b_operationally_valid_stage_c_not_ready"
            and readiness_contracts.get("stage_b_selected_and_baseline_disjoint") is True
            and readiness_contracts.get("indexed_all_pairs_equivalent") is True
            and bool(broad_readiness.get("stage_c_blockers"))
            and broad_readiness.get("utility_scope") == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_BROAD_READINESS_PATH),
                "status": broad_readiness.get("status"),
                "stage_c_blockers": broad_readiness.get("stage_c_blockers"),
            },
        )
    )
    broad_ablations = load_json(TEMPORAL_CODE_BROAD_ABLATIONS_PATH) if TEMPORAL_CODE_BROAD_ABLATIONS_PATH.exists() else {}
    items.append(
        ValidationItem(
            name="temporal_code_broad_frozen_stage_b_ablations",
            ok=broad_ablations.get("schema_version") == "temporal-code-broad-stage-b-ablations-v1"
            and set((broad_ablations.get("arms") or {}).keys())
            == {"full_selector", "quality_only", "redundancy_only", "no_coverage_support"}
            and broad_ablations.get("forbidden_signals_observed") == []
            and broad_ablations.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_BROAD_ABLATIONS_PATH), "arms": broad_ablations.get("arms")},
        )
    )
    path_contract = (
        load_json(TEMPORAL_CODE_PATH_STRATIFIED_CONTRACT_PATH)
        if TEMPORAL_CODE_PATH_STRATIFIED_CONTRACT_PATH.exists()
        else {}
    )
    path_forbidden = set(_as_str_list(path_contract.get("selection_forbids")))
    items.append(
        ValidationItem(
            name="temporal_code_path_stratified_contract",
            ok=path_contract.get("schema_version") == "temporal-code-path-stratified-tranche-v2"
            and path_contract.get("one_pull_request_per_repository") is True
            and {"file content", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
                path_forbidden
            )
            and path_contract.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_PATH_STRATIFIED_CONTRACT_PATH)},
        )
    )
    path_plan = (
        load_json(TEMPORAL_CODE_PATH_STRATIFIED_PLAN_PATH)
        if TEMPORAL_CODE_PATH_STRATIFIED_PLAN_PATH.exists()
        else {}
    )
    items.append(
        ValidationItem(
            name="temporal_code_path_stratified_plan",
            ok=path_plan.get("schema_version") == "temporal-code-path-stratified-tranche-plan-v1"
            and path_plan.get("status") == "frozen_before_tranche_content_fetch"
            and int((path_plan.get("summary") or {}).get("repository_count") or 0) == 40
            and not ((path_plan.get("summary") or {}).get("blockers") or [])
            and all(
                len(row.get("sampled_prs") or []) == 1
                for rows in (path_plan.get("selected_repositories") or {}).values()
                for row in rows
            ),
            details={"path": str(TEMPORAL_CODE_PATH_STRATIFIED_PLAN_PATH), "summary": path_plan.get("summary")},
        )
    )
    path_readiness = (
        load_json(TEMPORAL_CODE_PATH_STRATIFIED_READINESS_PATH)
        if TEMPORAL_CODE_PATH_STRATIFIED_READINESS_PATH.exists()
        else {}
    )
    path_blockers = set(_as_str_list(path_readiness.get("stage_c_blockers")))
    items.append(
        ValidationItem(
            name="temporal_code_path_stratified_readiness_boundary",
            ok=path_readiness.get("status") == "ready_for_stage_c_smoke"
            and not path_blockers
            and (path_readiness.get("summary") or {}).get("executable_evaluation_gate_pass_by_split")
            == {"train": 1, "development": 1, "confirmatory": 1}
            and path_readiness.get("utility_scope") == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_PATH_STRATIFIED_READINESS_PATH),
                "stage_c_blockers": sorted(path_blockers),
            },
        )
    )
    path_ablations = (
        load_json(TEMPORAL_CODE_PATH_STRATIFIED_ABLATIONS_PATH)
        if TEMPORAL_CODE_PATH_STRATIFIED_ABLATIONS_PATH.exists()
        else {}
    )
    path_arms = path_ablations.get("arms") or {}
    items.append(
        ValidationItem(
            name="temporal_code_path_stratified_stage_b_ablations",
            ok=set(path_arms) == {"full_selector", "quality_only", "redundancy_only", "no_coverage_support"}
            and path_arms.get("full_selector") != path_arms.get("quality_only")
            and path_ablations.get("forbidden_signals_observed") == []
            and path_ablations.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_PATH_STRATIFIED_ABLATIONS_PATH), "arms": path_arms},
        )
    )
    development_expansion = (
        load_json(TEMPORAL_CODE_DEVELOPMENT_EXPANSION_PLAN_PATH)
        if TEMPORAL_CODE_DEVELOPMENT_EXPANSION_PLAN_PATH.exists()
        else {}
    )
    development_expansion_rows = (
        (development_expansion.get("selected_repositories") or {}).get("development") or []
    )
    development_expansion_forbidden = set(
        _as_str_list((development_expansion.get("contract") or {}).get("selection_forbids"))
    )
    items.append(
        ValidationItem(
            name="temporal_code_development_execution_expansion",
            ok=development_expansion.get("status") == "frozen_before_tranche_content_fetch"
            and len(development_expansion_rows) >= 11
            and all(row.get("assigned_split") == "development" for row in development_expansion_rows)
            and all(len(row.get("sampled_prs") or []) == 1 for row in development_expansion_rows)
            and (development_expansion.get("summary") or {}).get("development_utility_remains_blocked") is True
            and {"test execution outcomes", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
                development_expansion_forbidden
            )
            and development_expansion.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_DEVELOPMENT_EXPANSION_PLAN_PATH)},
        )
    )
    development_expansion_readiness = (
        load_json(TEMPORAL_CODE_DEVELOPMENT_EXPANSION_READINESS_PATH)
        if TEMPORAL_CODE_DEVELOPMENT_EXPANSION_READINESS_PATH.exists()
        else {}
    )
    development_expansion_summary = development_expansion_readiness.get("summary") or {}
    development_expansion_decision = development_expansion_readiness.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_development_expansion_readiness",
            ok=development_expansion_readiness.get("status")
            == "development_stage_c_blocked_insufficient_executable_holdout"
            and development_expansion_summary.get("frozen_candidate_repositories") == 11
            and development_expansion_summary.get("generic_execution_verified_bundles") == 0
            and development_expansion_summary.get("total_verified_development_bundles") == 1
            and development_expansion_decision.get("development_utility_may_start") is False
            and development_expansion_readiness.get("confirmatory_outcomes_read") is False
            and development_expansion_readiness.get("utility_scope")
            == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_DEVELOPMENT_EXPANSION_READINESS_PATH),
                "summary": development_expansion_summary,
            },
        )
    )
    native_commands = (
        load_json(TEMPORAL_CODE_NATIVE_EXECUTION_COMMANDS_PATH)
        if TEMPORAL_CODE_NATIVE_EXECUTION_COMMANDS_PATH.exists()
        else {}
    )
    native_forbidden = set(_as_str_list(native_commands.get("forbidden_inputs")))
    items.append(
        ValidationItem(
            name="temporal_code_native_execution_recipe_contract",
            ok=native_commands.get("status") == "refrozen_before_second_native_execution"
            and (native_commands.get("summary") or {}).get("repository_count") == 11
            and (native_commands.get("summary") or {}).get("nondefault_python_image_count", 0) >= 1
            and all(
                row.get("generic_execution_outcomes_read") is False
                and row.get("writable_workspace_copy") is True
                for row in (native_commands.get("repository_commands") or {}).values()
            )
            and {"generic execution outcomes", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
                native_forbidden
            )
            and native_commands.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_NATIVE_EXECUTION_COMMANDS_PATH)},
        )
    )
    native_refinement = (
        load_json(TEMPORAL_CODE_NATIVE_EXECUTION_REFINEMENT_PATH)
        if TEMPORAL_CODE_NATIVE_EXECUTION_REFINEMENT_PATH.exists()
        else {}
    )
    native_refinement_summary = native_refinement.get("summary") or {}
    native_refinement_decision = native_refinement.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_native_execution_refinement_boundary",
            ok=native_refinement.get("status") == "native_recipe_exploration_no_executable_recovery"
            and native_refinement_summary.get("native_v1_build_pass_commits", 0)
            > native_refinement_summary.get("generic_build_pass_commits", 0)
            and native_refinement_summary.get("native_v2_verified_bundles") == 0
            and native_refinement_decision.get("development_utility_may_start") is False
            and native_refinement_decision.get("continue_recipe_tuning_on_same_development_pool") is False
            and native_refinement.get("confirmatory_outcomes_read") is False
            and native_refinement.get("utility_scope") == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_NATIVE_EXECUTION_REFINEMENT_PATH),
                "summary": native_refinement_summary,
            },
        )
    )
    fresh_development_plan = (
        load_json(TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_PLAN_PATH)
        if TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_PLAN_PATH.exists()
        else {}
    )
    fresh_development_rows = (
        (fresh_development_plan.get("selected_repositories") or {}).get("development") or []
    )
    fresh_development_forbidden = set(
        _as_str_list((fresh_development_plan.get("contract") or {}).get("selection_forbids"))
    )
    items.append(
        ValidationItem(
            name="temporal_code_development_fresh_expansion_contract",
            ok=fresh_development_plan.get("status") == "frozen_before_tranche_content_fetch"
            and len(fresh_development_rows) == 14
            and all(row.get("assigned_split") == "development" for row in fresh_development_rows)
            and all(row.get("path_stratum") in {"test_only", "code_only"} for row in fresh_development_rows)
            and {"generic execution outcomes", "native execution outcomes", "Utility", "benchmark outcomes"}.issubset(
                fresh_development_forbidden
            )
            and fresh_development_plan.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_PLAN_PATH)},
        )
    )
    fresh_development_report = (
        load_json(TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_REPORT_PATH)
        if TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_REPORT_PATH.exists()
        else {}
    )
    fresh_development_summary = fresh_development_report.get("summary") or {}
    fresh_development_decision = fresh_development_report.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_development_fresh_expansion_boundary",
            ok=fresh_development_report.get("status") == "raw_repository_execution_support_insufficient"
            and fresh_development_summary.get("collection_gate_pass_bundles") == 12
            and fresh_development_summary.get("native_build_pass_commits", 0)
            >= fresh_development_summary.get("generic_build_pass_commits", 0)
            and fresh_development_summary.get("native_verified_bundles") == 0
            and fresh_development_decision.get("development_utility_may_start") is False
            and fresh_development_decision.get("broaden_raw_repository_discovery_for_execution_recovery") is False
            and fresh_development_report.get("confirmatory_outcomes_read") is False
            and fresh_development_report.get("utility_scope") == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_DEVELOPMENT_FRESH_EXPANSION_REPORT_PATH),
                "summary": fresh_development_summary,
            },
        )
    )
    execution_support = (
        load_json(TEMPORAL_CODE_EXECUTION_SUPPORT_REPORT_PATH)
        if TEMPORAL_CODE_EXECUTION_SUPPORT_REPORT_PATH.exists()
        else {}
    )
    execution_support_contract = execution_support.get("contract") or {}
    execution_support_rules = execution_support_contract.get("stage_entry_rules") or {}
    execution_support_summary = execution_support.get("summary") or {}
    execution_support_decision = execution_support.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_execution_support_tiers",
            ok=execution_support.get("status") == "orthogonal_content_and_execution_tiers_operational"
            and set((execution_support_contract.get("orthogonal_axes") or {}))
            == {"training_content", "execution_support"}
            and execution_support_rules.get("stage_b_selector") == "must not use execution tier"
            and execution_support_rules.get("utility") == "Stage C validation only; never selector objective"
            and execution_support_summary.get("training_content_eligible_count", 0)
            > execution_support_summary.get("executable_stage_c_eligible_count", 0)
            and execution_support_decision.get("training_content_may_be_preserved_without_executable_support")
            is True
            and execution_support_decision.get("execution_tier_may_enter_stage_b") is False
            and execution_support_decision.get("development_utility_may_start") is False
            and execution_support.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_EXECUTION_SUPPORT_REPORT_PATH),
                "summary": execution_support_summary,
            },
        )
    )
    executable_harness = (
        load_json(TEMPORAL_CODE_EXECUTABLE_TASK_HARNESS_PLAN_PATH)
        if TEMPORAL_CODE_EXECUTABLE_TASK_HARNESS_PLAN_PATH.exists()
        else {}
    )
    executable_harness_contract = executable_harness.get("contract") or {}
    executable_harness_eligibility = executable_harness_contract.get("eligibility") or {}
    executable_harness_sample_size = executable_harness_contract.get("sample_size_rule") or {}
    executable_harness_forbidden = set(_as_str_list(executable_harness_contract.get("forbidden_uses")))
    items.append(
        ValidationItem(
            name="temporal_code_executable_task_harness_contract",
            ok=executable_harness.get("status") == "frozen_contract_source_profiled_e2_acquisition_blocked"
            and executable_harness_contract.get("task_role") == "evaluation_only_never_training"
            and executable_harness_eligibility.get("execution_support_tier") == "E2"
            and executable_harness_sample_size.get("fixed_arbitrary_minimum_forbidden") is True
            and executable_harness_sample_size.get("practical_effect_margin_absolute") == 0.05
            and executable_harness_sample_size.get("training_seed_count") == 5
            and {"using task outcomes in Stage B", "using different Stage-A baselines for sensitivity arms"}.issubset(
                executable_harness_forbidden
            )
            and (executable_harness.get("current_evidence") or {}).get("development_utility_may_start") is False
            and executable_harness.get("confirmatory_outcomes_read") is False
            and executable_harness.get("utility_scope") == "Stage C validation only; never selector objective",
            details={
                "path": str(TEMPORAL_CODE_EXECUTABLE_TASK_HARNESS_PLAN_PATH),
                "entry_blockers": executable_harness.get("entry_blockers") or [],
            },
        )
    )
    swebench_profile = (
        load_json(TEMPORAL_CODE_SWEBENCH_HARNESS_METADATA_PROFILE_PATH)
        if TEMPORAL_CODE_SWEBENCH_HARNESS_METADATA_PROFILE_PATH.exists()
        else {}
    )
    swebench_split = swebench_profile.get("split_summary") or {}
    swebench_precision = swebench_profile.get("precision_analysis") or {}
    swebench_e2 = swebench_profile.get("e2_analysis") or {}
    items.append(
        ValidationItem(
            name="temporal_code_swebench_harness_metadata_profile",
            ok=swebench_profile.get("status") == "outcome_free_source_profile_complete"
            and (swebench_profile.get("source_summary") or {}).get("raw_task_content_persisted") is False
            and (swebench_profile.get("source_summary") or {}).get("model_outcomes_read") is False
            and swebench_split.get("repository_overlap_count") == 0
            and swebench_precision.get("required_task_count") == 1083
            and swebench_precision.get("eligible_count_meets_required_task_count") is False
            and swebench_e2.get("e2_verified_task_count") == 0
            and swebench_profile.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_SWEBENCH_HARNESS_METADATA_PROFILE_PATH),
                "split_summary": swebench_split,
                "precision_analysis": swebench_precision,
            },
        )
    )
    evalplus_guardrail = (
        load_json(TEMPORAL_CODE_EVALPLUS_GUARDRAIL_PREVALIDATION_PATH)
        if TEMPORAL_CODE_EVALPLUS_GUARDRAIL_PREVALIDATION_PATH.exists()
        else {}
    )
    evalplus_environment = evalplus_guardrail.get("environment") or {}
    evalplus_decision = evalplus_guardrail.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_evalplus_guardrail_prevalidation",
            ok=evalplus_guardrail.get("status") == "e2_prevalidated"
            and evalplus_environment.get("resource_module_available") is False
            and evalplus_environment.get("docker_daemon_available") is True
            and evalplus_environment.get("isolated_backend") == "docker_linux"
            and bool(evalplus_environment.get("isolated_image_id"))
            and evalplus_environment.get("model_generated_code_executed") is False
            and evalplus_decision.get("semantic_controls_executed") is True
            and evalplus_decision.get("semantic_controls_pass") is True
            and evalplus_decision.get("execution_support_tier") == "E2"
            and evalplus_decision.get("may_enter_stage_c_guardrail") is True
            and evalplus_decision.get("may_replace_primary_temporal_executable_aggregate") is False
            and evalplus_guardrail.get("task_content_persisted") is False
            and evalplus_guardrail.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_EVALPLUS_GUARDRAIL_PREVALIDATION_PATH),
                "environment": evalplus_environment,
                "decision": evalplus_decision,
            },
        )
    )
    evalplus_split = (
        load_json(TEMPORAL_CODE_EVALPLUS_GUARDRAIL_SPLIT_PATH)
        if TEMPORAL_CODE_EVALPLUS_GUARDRAIL_SPLIT_PATH.exists()
        else {}
    )
    evalplus_split_summary = evalplus_split.get("summary") or {}
    evalplus_split_contract = evalplus_split.get("contract") or {}
    items.append(
        ValidationItem(
            name="temporal_code_evalplus_guardrail_split",
            ok=evalplus_split.get("status") == "frozen_e2_guardrail_split_before_model_outcomes"
            and evalplus_split_summary.get("task_count") == 542
            and set((evalplus_split_summary.get("split_counts") or {})) == {"development", "confirmatory"}
            and evalplus_split_summary.get("task_content_persisted") is False
            and evalplus_split_summary.get("model_outcomes_read") is False
            and (evalplus_split_contract.get("non_inferiority") or {}).get(
                "maximum_allowed_absolute_regression_macro"
            )
            == 0.02
            and evalplus_split.get("development_utility_may_start") is False
            and evalplus_split.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_EVALPLUS_GUARDRAIL_SPLIT_PATH),
                "summary": evalplus_split_summary,
            },
        )
    )
    retention_guardrail = (
        load_json(TEMPORAL_CODE_RETENTION_GUARDRAIL_PLAN_PATH)
        if TEMPORAL_CODE_RETENTION_GUARDRAIL_PLAN_PATH.exists()
        else {}
    )
    retention_contract = retention_guardrail.get("contract") or {}
    items.append(
        ValidationItem(
            name="temporal_code_retention_guardrail_contract",
            ok=retention_guardrail.get("status") == "frozen_before_development_model_outcomes"
            and (retention_contract.get("code_guardrail") or {}).get(
                "maximum_allowed_absolute_regression_macro"
            )
            == 0.02
            and (retention_contract.get("general_task_guardrail") or {}).get(
                "maximum_allowed_absolute_regression_macro"
            )
            == 0.01
            and (retention_contract.get("general_text_guardrail") or {}).get(
                "maximum_allowed_mean_nll_increase"
            )
            == 0.01
            and (retention_contract.get("decision_rule") or {}).get("all_guardrails_mandatory") is True
            and retention_guardrail.get("development_utility_may_start") is False
            and retention_guardrail.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_RETENTION_GUARDRAIL_PLAN_PATH),
                "remaining_blockers": retention_guardrail.get("remaining_blockers") or [],
            },
        )
    )
    primary_source = (
        load_json(TEMPORAL_CODE_PRIMARY_SOURCE_ASSESSMENT_PATH)
        if TEMPORAL_CODE_PRIMARY_SOURCE_ASSESSMENT_PATH.exists()
        else {}
    )
    primary_source_summary = primary_source.get("summary") or {}
    primary_source_decision = primary_source.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_primary_executable_source_assessment",
            ok=primary_source.get("status")
            == "primary_temporal_executable_distribution_not_currently_acquirable_from_frozen_sources"
            and primary_source_summary.get("required_primary_task_count") == 1083
            and primary_source_summary.get("current_primary_temporal_e2_task_count") == 2
            and primary_source_summary.get("task_count_gap") == 1081
            and primary_source_summary.get("evalplus_e2_guardrail_frozen") is True
            and primary_source_summary.get("current_public_source_meets_primary_contract") is False
            and primary_source_decision.get("development_utility_may_start") is False
            and primary_source_decision.get("retroactive_contract_weakening_allowed") is False
            and primary_source.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_PRIMARY_SOURCE_ASSESSMENT_PATH),
                "summary": primary_source_summary,
                "decision": primary_source_decision,
            },
        )
    )
    forward_pilot = (
        load_json(TEMPORAL_CODE_FORWARD_E2_PILOT_REPORT_PATH)
        if TEMPORAL_CODE_FORWARD_E2_PILOT_REPORT_PATH.exists()
        else {}
    )
    forward_pilot_summary = forward_pilot.get("summary") or {}
    forward_pilot_decisions = forward_pilot.get("decisions") or []
    items.append(
        ValidationItem(
            name="temporal_code_forward_e2_pilot",
            ok=forward_pilot.get("status") == "forward_e2_infrastructure_pilot_complete"
            and forward_pilot_summary.get("metadata_candidate_count") == 16
            and forward_pilot_summary.get("execution_candidate_count") == 5
            and forward_pilot_summary.get("task_valid_e2_count") == 2
            and forward_pilot_summary.get("pilot_tasks_evaluation_authorized_count") == 0
            and all(row.get("pilot_task_evaluation_authorized") is False for row in forward_pilot_decisions)
            and all("failure_stage" in row for row in forward_pilot_decisions)
            and (forward_pilot.get("decision") or {}).get("development_utility_may_start") is False
            and forward_pilot.get("confirmatory_outcomes_read") is False,
            details={"path": str(TEMPORAL_CODE_FORWARD_E2_PILOT_REPORT_PATH), "summary": forward_pilot_summary},
        )
    )
    forward_productivity = (
        load_json(TEMPORAL_CODE_FORWARD_E2_PRODUCTIVITY_REPORT_PATH)
        if TEMPORAL_CODE_FORWARD_E2_PRODUCTIVITY_REPORT_PATH.exists()
        else {}
    )
    forward_estimates = forward_productivity.get("point_estimate_only") or {}
    forward_interpretation = forward_productivity.get("interpretation") or {}
    items.append(
        ValidationItem(
            name="temporal_code_forward_e2_productivity",
            ok=forward_productivity.get("status") == "forward_e2_acquisition_feasible_but_not_ready_for_utility"
            and forward_estimates.get("metadata_candidates_needed_for_1083") == 8664
            and forward_estimates.get("execution_attempts_needed_for_1083") == 2708
            and forward_interpretation.get("pilot_tasks_evaluation_authorized") is False
            and forward_interpretation.get("pilot_too_small_for_capacity_commitment") is True
            and forward_interpretation.get("inferential_yield_or_capacity_claim_allowed") is False
            and forward_interpretation.get("development_utility_may_start") is False
            and forward_interpretation.get("confirmatory_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_FORWARD_E2_PRODUCTIVITY_REPORT_PATH),
                "point_estimate_only": forward_estimates,
                "interpretation": forward_interpretation,
            },
        )
    )
    forward_development = (
        load_json(TEMPORAL_CODE_FORWARD_DEVELOPMENT_SNAPSHOT_REPORT_PATH)
        if TEMPORAL_CODE_FORWARD_DEVELOPMENT_SNAPSHOT_REPORT_PATH.exists()
        else {}
    )
    forward_development_summary = forward_development.get("summary") or {}
    forward_development_decision = forward_development.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_forward_development_snapshot",
            ok=forward_development.get("status") == "forward_development_snapshot_complete_no_candidates"
            and forward_development_summary.get("fresh_repository_frame_count") == 200
            and forward_development_summary.get("metadata_candidate_count") == 0
            and forward_development_summary.get("training_repository_overlap_count") == 0
            and forward_development_summary.get("execution_recipe_count") == 0
            and forward_development_summary.get("task_valid_e2_count") == 0
            and forward_development_decision.get("zero_candidates_is_valid_snapshot_evidence") is True
            and forward_development_decision.get("retroactively_expand_same_snapshot_after_candidate_outcome") is False
            and forward_development_decision.get("candidate_recipe_or_execution_may_start") is False
            and forward_development_decision.get("development_utility_may_start") is False
            and forward_development_decision.get("confirmatory_outcomes_read") is False
            and forward_development.get("execution_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_FORWARD_DEVELOPMENT_SNAPSHOT_REPORT_PATH),
                "summary": forward_development_summary,
                "decision": forward_development_decision,
            },
        )
    )
    forward_accumulation = (
        load_json(TEMPORAL_CODE_FORWARD_DEVELOPMENT_ACCUMULATION_PLAN_PATH)
        if TEMPORAL_CODE_FORWARD_DEVELOPMENT_ACCUMULATION_PLAN_PATH.exists()
        else {}
    )
    accumulation_frame = forward_accumulation.get("accumulation_frame") or {}
    accumulation_capacity = forward_accumulation.get("capacity_context") or {}
    accumulation_contract = (forward_accumulation.get("contract") or {}).get(
        "development_accumulation_amendment"
    ) or {}
    items.append(
        ValidationItem(
            name="temporal_code_forward_development_accumulation",
            ok=forward_accumulation.get("status")
            == "frozen_after_snapshot_001_metadata_and_before_any_later_snapshot_metadata"
            and accumulation_frame.get("repository_count") == 5000
            and accumulation_frame.get("existing_broad_repository_overlap_count") == 0
            and accumulation_frame.get("benchmark_source_repository_overlap_count") == 0
            and accumulation_capacity.get("point_estimate_metadata_candidates_needed_for_development") == 4336
            and accumulation_capacity.get("frame_meets_point_estimate_candidate_capacity") is True
            and accumulation_capacity.get("frame_alone_guarantees_target") is False
            and accumulation_capacity.get("estimate_role") == "planning_only"
            and accumulation_contract.get("eligibility_rule_changes") == "none"
            and accumulation_contract.get("not_justified_by") == "snapshot_001_zero_candidate_outcome"
            and forward_accumulation.get("next_snapshot_task_metadata_read") is False
            and forward_accumulation.get("execution_outcomes_read") is False
            and forward_accumulation.get("confirmatory_outcomes_read") is False
            and forward_accumulation.get("development_utility_may_start") is False,
            details={
                "path": str(TEMPORAL_CODE_FORWARD_DEVELOPMENT_ACCUMULATION_PLAN_PATH),
                "accumulation_frame": accumulation_frame,
                "capacity_context": accumulation_capacity,
            },
        )
    )
    forward_capacity = (
        load_json(TEMPORAL_CODE_FORWARD_DISCOVERY_CAPACITY_REPORT_PATH)
        if TEMPORAL_CODE_FORWARD_DISCOVERY_CAPACITY_REPORT_PATH.exists()
        else {}
    )
    forward_capacity_summary = forward_capacity.get("summary") or {}
    forward_capacity_decision = forward_capacity.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_forward_discovery_capacity",
            ok=forward_capacity.get("status") == "forward_repository_frame_meets_point_estimate_candidate_capacity"
            and forward_capacity_summary.get("combined_discovered_repository_count") == 12067
            and forward_capacity_summary.get("frozen_fresh_repository_frame_count") == 5000
            and forward_capacity_summary.get("point_estimate_metadata_candidates_needed") == 4336
            and forward_capacity_summary.get("point_estimate_candidate_capacity_met") is True
            and forward_capacity_summary.get("actual_task_candidate_count") == 0
            and forward_capacity_summary.get("actual_e2_count") == 0
            and forward_capacity_decision.get("structural_repository_frame_blocker_resolved") is True
            and forward_capacity_decision.get("actual_task_distribution_blocker_resolved") is False
            and forward_capacity_decision.get("frame_guarantees_task_target") is False
            and forward_capacity_decision.get("development_utility_may_start") is False
            and forward_capacity_decision.get("confirmatory_outcomes_read") is False
            and forward_capacity.get("task_metadata_read_from_expanded_frame") is False
            and forward_capacity.get("execution_outcomes_read") is False,
            details={
                "path": str(TEMPORAL_CODE_FORWARD_DISCOVERY_CAPACITY_REPORT_PATH),
                "summary": forward_capacity_summary,
                "decision": forward_capacity_decision,
            },
        )
    )
    forward_schedule = (
        load_json(TEMPORAL_CODE_FORWARD_COLLECTION_SCHEDULE_PATH)
        if TEMPORAL_CODE_FORWARD_COLLECTION_SCHEDULE_PATH.exists()
        else {}
    )
    forward_ledger = (
        load_json(TEMPORAL_CODE_FORWARD_CANDIDATE_LEDGER_PATH)
        if TEMPORAL_CODE_FORWARD_CANDIDATE_LEDGER_PATH.exists()
        else {}
    )
    forward_operations = (
        load_json(TEMPORAL_CODE_FORWARD_OPERATIONS_STATUS_PATH)
        if TEMPORAL_CODE_FORWARD_OPERATIONS_STATUS_PATH.exists()
        else {}
    )
    schedule_summary = forward_schedule.get("summary") or {}
    ledger_summary = forward_ledger.get("summary") or {}
    operations_summary = forward_operations.get("summary") or {}
    operations_gates = forward_operations.get("gates") or {}
    items.append(
        ValidationItem(
            name="temporal_code_forward_operations",
            ok=forward_schedule.get("status") == "frozen_before_later_snapshot_task_metadata"
            and schedule_summary.get("repository_count") == 5000
            and schedule_summary.get("shard_size") == 200
            and schedule_summary.get("shard_count") == 25
            and schedule_summary.get("duplicate_repository_count") == 0
            and forward_ledger.get("status") == "candidate_ledger_frozen_before_recipe_or_execution"
            and ledger_summary.get("candidate_count") == 0
            and forward_operations.get("status") == "forward_collection_operational_waiting_for_later_date_tasks"
            and operations_summary.get("candidate_count") == 0
            and operations_summary.get("candidate_gap") == 542
            and operations_gates.get("repository_schedule_frozen") is True
            and operations_gates.get("snapshot_artifacts_immutable") is True
            and operations_gates.get("candidate_ledger_frozen_before_recipe") is True
            and operations_gates.get("recipe_freeze_may_start") is False
            and operations_gates.get("e2_execution_may_start") is False
            and operations_gates.get("development_utility_may_start") is False
            and operations_gates.get("confirmatory_outcomes_read") is False,
            details={
                "schedule": schedule_summary,
                "ledger": ledger_summary,
                "operations": operations_summary,
                "gates": operations_gates,
            },
        )
    )
    retrospective = (
        load_json(TEMPORAL_CODE_RETROSPECTIVE_DEVELOPMENT_REPORT_PATH)
        if TEMPORAL_CODE_RETROSPECTIVE_DEVELOPMENT_REPORT_PATH.exists()
        else {}
    )
    retrospective_expansion = (
        load_json(TEMPORAL_CODE_RETROSPECTIVE_EXPANSION_SCHEDULE_PATH)
        if TEMPORAL_CODE_RETROSPECTIVE_EXPANSION_SCHEDULE_PATH.exists()
        else {}
    )
    retrospective_ledger = (
        load_json(TEMPORAL_CODE_RETROSPECTIVE_COMBINED_LEDGER_PATH)
        if TEMPORAL_CODE_RETROSPECTIVE_COMBINED_LEDGER_PATH.exists()
        else {}
    )
    retrospective_status = (
        load_json(TEMPORAL_CODE_RETROSPECTIVE_OPERATIONS_STATUS_PATH)
        if TEMPORAL_CODE_RETROSPECTIVE_OPERATIONS_STATUS_PATH.exists()
        else {}
    )
    retrospective_capacity = (
        load_json(TEMPORAL_CODE_RETROSPECTIVE_E2_CAPACITY_AUDIT_PATH)
        if TEMPORAL_CODE_RETROSPECTIVE_E2_CAPACITY_AUDIT_PATH.exists()
        else {}
    )
    retrospective_observed = retrospective.get("observed") or {}
    retrospective_planning = retrospective.get("planning_estimate_only") or {}
    retrospective_decision = retrospective.get("decision") or {}
    retrospective_expansion_summary = retrospective_expansion.get("summary") or {}
    retrospective_adaptation = retrospective_expansion.get("adaptation_contract") or {}
    retrospective_ledger_summary = retrospective_ledger.get("summary") or {}
    retrospective_status_summary = retrospective_status.get("summary") or {}
    retrospective_status_gates = retrospective_status.get("gates") or {}
    retrospective_capacity_observed = retrospective_capacity.get("observed") or {}
    retrospective_capacity_decision = retrospective_capacity.get("decision") or {}
    items.append(
        ValidationItem(
            name="temporal_code_retrospective_development",
            ok=retrospective.get("status")
            == "retrospective_development_feasible_expansion_required_before_utility"
            and retrospective_observed.get("repositories_scanned") == 5000
            and retrospective_observed.get("strict_metadata_candidates") == 1666
            and retrospective_observed.get("first_e2_batch_execution_attempts") == 25
            and retrospective_observed.get("first_e2_batch_task_valid_count") == 4
            and retrospective_observed.get("training_repository_overlap_count") == 0
            and retrospective_planning.get("development_valid_e2_target") == 542
            and retrospective_planning.get("remaining_unscanned_repository_count") == 6822
            and retrospective_planning.get("remaining_frame_training_repository_exclusion_count") == 245
            and retrospective_planning.get("inferential_capacity_claim_allowed") is False
            and retrospective_decision.get("actual_task_distribution_blocker_resolved") is False
            and retrospective_decision.get("same_rules_full_remaining_frame_expansion_justified") is True
            and retrospective_decision.get("task_validity_rule_may_be_weakened") is False
            and retrospective_decision.get("development_utility_may_start") is False
            and retrospective_decision.get("confirmatory_outcomes_read") is False
            and retrospective_expansion.get("status")
            == "frozen_after_first_e2_batch_and_before_remaining_repository_task_metadata"
            and retrospective_expansion_summary.get("remaining_repository_count") == 6822
            and retrospective_expansion_summary.get("shard_count") == 35
            and retrospective_expansion_summary.get("initial_repository_overlap_count") == 0
            and retrospective_expansion_summary.get("training_repository_overlap_count") == 0
            and retrospective_adaptation.get("eligibility_rule_changes") == "none"
            and retrospective_adaptation.get("task_validity_rule_changes") == "none"
            and retrospective_expansion.get("confirmatory_outcomes_read") is False
            and retrospective_expansion.get("development_utility_may_start") is False
            and retrospective_ledger.get("status")
            == "combined_retrospective_candidate_ledger_frozen_before_recipe_or_execution"
            and retrospective_ledger_summary.get("scheduled_repository_count") == 11822
            and retrospective_ledger_summary.get("snapshot_count") == 60
            and retrospective_ledger_summary.get("expected_snapshot_count") == 60
            and retrospective_ledger_summary.get("metadata_collection_complete") is True
            and retrospective_ledger_summary.get("candidate_count") == 3847
            and retrospective_status.get("status") == "retrospective_collection_ready_for_remaining_e2_batches"
            and retrospective_status_summary.get("e2_execution_attempt_count") == 825
            and retrospective_status_summary.get("task_valid_e2_count") == 167
            and retrospective_status_summary.get("valid_e2_gap") == 375
            and retrospective_status_gates.get("metadata_collection_complete") is True
            and retrospective_status_gates.get("development_utility_may_start") is False
            and retrospective_status_gates.get("confirmatory_outcomes_read") is False
            and retrospective_capacity.get("status") == "retrospective_strict_e2_execution_should_continue"
            and retrospective_capacity_observed.get("execution_attempt_count") == 825
            and retrospective_capacity_observed.get("task_valid_e2_count") == 167
            and retrospective_capacity_decision.get("strict_e2_execution_may_continue") is True
            and retrospective_capacity_decision.get("development_utility_may_start") is False,
            details={
                "observed": retrospective_observed,
                "planning_estimate_only": retrospective_planning,
                "decision": retrospective_decision,
                "expansion_summary": retrospective_expansion_summary,
                "ledger_summary": retrospective_ledger_summary,
                "operations_summary": retrospective_status_summary,
                "operations_gates": retrospective_status_gates,
                "capacity_observed": retrospective_capacity_observed,
                "capacity_decision": retrospective_capacity_decision,
            },
        )
    )
    confirmatory_expansion = (
        load_json(TEMPORAL_CODE_CONFIRMATORY_EXPANSION_PLAN_PATH)
        if TEMPORAL_CODE_CONFIRMATORY_EXPANSION_PLAN_PATH.exists()
        else {}
    )
    expansion_rows = (confirmatory_expansion.get("selected_repositories") or {}).get("confirmatory") or []
    expansion_forbidden = set(
        _as_str_list((confirmatory_expansion.get("contract") or {}).get("selection_forbids"))
    )
    items.append(
        ValidationItem(
            name="temporal_code_confirmatory_execution_expansion",
            ok=confirmatory_expansion.get("status") == "frozen_before_tranche_content_fetch"
            and len(expansion_rows) == 12
            and all(row.get("assigned_split") == "confirmatory" for row in expansion_rows)
            and all(len(row.get("sampled_prs") or []) == 1 for row in expansion_rows)
            and {"test execution outcomes", "Utility", "benchmark outcomes", "human or LLM review labels"}.issubset(
                expansion_forbidden
            ),
            details={"path": str(TEMPORAL_CODE_CONFIRMATORY_EXPANSION_PLAN_PATH)},
        )
    )
    stage_c_smoke_contract = (
        load_json(TEMPORAL_CODE_STAGE_C_SMOKE_CONTRACT_PATH)
        if TEMPORAL_CODE_STAGE_C_SMOKE_CONTRACT_PATH.exists()
        else {}
    )
    stage_c_arm_contract = stage_c_smoke_contract.get("arm_contract") or {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_c_smoke_contract",
            ok=stage_c_smoke_contract.get("status") == "frozen_before_target_tokenization_or_model_execution"
            and (stage_c_smoke_contract.get("target_model") or {}).get("model_id") == "Qwen/Qwen3-4B-Base"
            and stage_c_arm_contract.get("all_sensitivity_arms_must_share_common_stage_a_baseline") is True
            and stage_c_arm_contract.get("common_stage_a_baseline_must_be_disjoint_from_every_sensitivity_arm")
            is True
            and stage_c_smoke_contract.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_C_SMOKE_CONTRACT_PATH)},
        )
    )
    stage_c_arm_manifest = (
        load_json(TEMPORAL_CODE_STAGE_C_SMOKE_ARM_MANIFEST_PATH)
        if TEMPORAL_CODE_STAGE_C_SMOKE_ARM_MANIFEST_PATH.exists()
        else {}
    )
    common_baselines = stage_c_arm_manifest.get("sensitivity_common_stage_a_baseline_sha256") or {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_c_target_token_arms",
            ok=stage_c_arm_manifest.get("status") == "frozen_target_token_arms_before_model_execution"
            and stage_c_arm_manifest.get("curated_common_baseline_overlap_count") == 0
            and stage_c_arm_manifest.get("all_sensitivity_arms_share_common_stage_a_baseline") is True
            and len(set(common_baselines.values())) == 1
            and stage_c_arm_manifest.get("confirmatory_content_read") is False
            and stage_c_arm_manifest.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_C_SMOKE_ARM_MANIFEST_PATH)},
        )
    )
    stage_c_blocks = (
        load_json(TEMPORAL_CODE_STAGE_C_SMOKE_BLOCK_MANIFEST_PATH)
        if TEMPORAL_CODE_STAGE_C_SMOKE_BLOCK_MANIFEST_PATH.exists()
        else {}
    )
    packed_tokens = {
        int(row.get("packed_tokens") or 0) for row in (stage_c_blocks.get("blocks") or {}).values()
    }
    items.append(
        ValidationItem(
            name="temporal_code_stage_c_equal_packed_token_blocks",
            ok=stage_c_blocks.get("status") == "frozen_equal_packed_token_blocks"
            and len(packed_tokens) == 1
            and next(iter(packed_tokens), 0) > 0
            and stage_c_blocks.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_C_SMOKE_BLOCK_MANIFEST_PATH), "packed_tokens": sorted(packed_tokens)},
        )
    )
    stage_c_smoke_report = (
        load_json(TEMPORAL_CODE_STAGE_C_SMOKE_REPORT_PATH)
        if TEMPORAL_CODE_STAGE_C_SMOKE_REPORT_PATH.exists()
        else {}
    )
    stage_c_smoke_summary = stage_c_smoke_report.get("summary") or {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_c_qlora_smoke_feasibility",
            ok=stage_c_smoke_report.get("status") == "qlora_stage_c_smoke_feasibility_pass"
            and stage_c_smoke_summary.get("all_arms_completed") is True
            and stage_c_smoke_summary.get("equal_packed_token_budget") is True
            and stage_c_smoke_summary.get("equal_optimizer_steps") is True
            and stage_c_smoke_summary.get("equal_seed") is True
            and stage_c_smoke_summary.get("common_stage_a_baseline_shared") is True
            and stage_c_smoke_summary.get("curated_common_baseline_overlap_count") == 0
            and stage_c_smoke_report.get("confirmatory_outcomes_read") is False
            and "not Utility" in stage_c_smoke_report.get("training_loss_interpretation", "")
            and stage_c_smoke_report.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_C_SMOKE_REPORT_PATH), "summary": stage_c_smoke_summary},
        )
    )
    proxy_report = load_json(TEMPORAL_CODE_STAGE_B_PROXY_VALIDATION_PATH) if TEMPORAL_CODE_STAGE_B_PROXY_VALIDATION_PATH.exists() else {}
    proxy_summary = proxy_report.get("summary") if isinstance(proxy_report.get("summary"), dict) else {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_proxy_validation",
            ok=proxy_report.get("schema_version") == "temporal-code-stage-b-proxy-validation-v1"
            and int(proxy_summary.get("assertion_count") or 0) >= 7
            and int(proxy_summary.get("failed_count", -1)) == 0
            and proxy_report.get("utility_scope") == "Stage C validation only; never selector objective",
            details={"path": str(TEMPORAL_CODE_STAGE_B_PROXY_VALIDATION_PATH), "summary": proxy_summary},
        )
    )
    index_report = load_json(TEMPORAL_CODE_STAGE_B_INDEX_EQUIVALENCE_PATH) if TEMPORAL_CODE_STAGE_B_INDEX_EQUIVALENCE_PATH.exists() else {}
    index_summary = index_report.get("summary") if isinstance(index_report.get("summary"), dict) else {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_index_equivalence",
            ok=index_report.get("schema_version") == "temporal-code-stage-b-index-equivalence-v1"
            and index_summary.get("passed") is True
            and int(index_summary.get("selected_symmetric_difference_count", -1)) == 0
            and int(index_summary.get("baseline_symmetric_difference_count", -1)) == 0,
            details={"path": str(TEMPORAL_CODE_STAGE_B_INDEX_EQUIVALENCE_PATH), "summary": index_summary},
        )
    )
    packet = load_json(TEMPORAL_CODE_STAGE_B_BLIND_PACKET_PATH) if TEMPORAL_CODE_STAGE_B_BLIND_PACKET_PATH.exists() else {}
    key = load_json(TEMPORAL_CODE_STAGE_B_BLIND_KEY_PATH) if TEMPORAL_CODE_STAGE_B_BLIND_KEY_PATH.exists() else {}
    packet_records = packet.get("records") if isinstance(packet.get("records"), list) else []
    key_records = key.get("records") if isinstance(key.get("records"), list) else []
    forbidden_packet_fields = {"chunk_uid", "repository_identity", "bundle_id", "path", "arm", "stage_b_evidence", "sampling_stratum", "stratum"}
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_optional_blind_review_diagnostic",
            ok=not packet
            or (
                packet.get("schema_version") == "temporal-code-stage-b-blind-review-packet-v1"
                and packet.get("status") == "awaiting_independent_review"
                and len(packet_records) >= 60
                and len(packet_records) == len(key_records)
                and all(not forbidden_packet_fields.intersection(row) for row in packet_records)
                and (packet.get("review_contract") or {}).get("scores_and_selection_arms_hidden") is True
                and (packet.get("review_contract") or {}).get("sampling_strata_hidden") is True
            ),
            details={
                "required_for_stage_b_or_stage_c_entry": False,
                "packet_path": str(TEMPORAL_CODE_STAGE_B_BLIND_PACKET_PATH),
                "review_record_count": len(packet_records),
            },
        )
    )
    expansion = load_json(TEMPORAL_CODE_PROXY_REVIEW_EXPANSION_PATH) if TEMPORAL_CODE_PROXY_REVIEW_EXPANSION_PATH.exists() else {}
    expansion_summary = expansion.get("summary") if isinstance(expansion.get("summary"), dict) else {}
    expansion_boundary = expansion.get("review_only_boundary") if isinstance(expansion.get("review_only_boundary"), dict) else {}
    reviewed_repositories = {row.get("repository_identity") for row in key_records}
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_optional_review_corpus_scope",
            ok=int(expansion_summary.get("combined_repository_count") or 0) >= 3
            and (not key_records or len(reviewed_repositories) >= 3)
            and expansion_boundary.get("training_approval") is False
            and expansion_boundary.get("stage0_release_candidate") is False
            and expansion_boundary.get("test_command_verified") is False,
            details={
                "required_for_stage_b_or_stage_c_entry": False,
                "expansion_summary": expansion_summary,
                "reviewed_repositories": sorted(str(value) for value in reviewed_repositories),
            },
        )
    )
    blind_analysis = load_json(TEMPORAL_CODE_STAGE_B_BLIND_ANALYSIS_PATH) if TEMPORAL_CODE_STAGE_B_BLIND_ANALYSIS_PATH.exists() else {}
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_optional_review_cannot_promote_proxy",
            ok=not blind_analysis
            or (
                blind_analysis.get("status") in {
                    "blocked_incomplete_independent_review",
                    "independent_review_complete_initial_real_corpus_evidence",
                }
                and blind_analysis.get("proxy_promotion_allowed") is False
            ),
            details={
                "required_for_stage_b_or_stage_c_entry": False,
                "path": str(TEMPORAL_CODE_STAGE_B_BLIND_ANALYSIS_PATH),
                "status": blind_analysis.get("status"),
            },
        )
    )
    reviewer_a = load_json(TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_DIR / "reviewer_a_packet.json") if (TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_DIR / "reviewer_a_packet.json").exists() else {}
    reviewer_b = load_json(TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_DIR / "reviewer_b_packet.json") if (TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_DIR / "reviewer_b_packet.json").exists() else {}
    multi_analysis = load_json(TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_ANALYSIS_PATH) if TEMPORAL_CODE_STAGE_B_MULTI_REVIEW_ANALYSIS_PATH.exists() else {}
    reviewer_a_records = reviewer_a.get("records") if isinstance(reviewer_a.get("records"), list) else []
    reviewer_b_records = reviewer_b.get("records") if isinstance(reviewer_b.get("records"), list) else []
    reviewer_a_ids = [row.get("review_id") for row in reviewer_a_records]
    reviewer_b_ids = [row.get("review_id") for row in reviewer_b_records]
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_optional_multi_review_contract",
            ok=not reviewer_a
            or (
                reviewer_a.get("schema_version") == "temporal-code-stage-b-independent-reviewer-packet-v1"
                and reviewer_b.get("schema_version") == "temporal-code-stage-b-independent-reviewer-packet-v1"
                and len(reviewer_a_records) == len(packet_records)
                and set(reviewer_a_ids) == set(reviewer_b_ids) == {row.get("review_id") for row in packet_records}
                and reviewer_a_ids != reviewer_b_ids
                and all(not forbidden_packet_fields.intersection(row) for row in reviewer_a_records + reviewer_b_records)
                and multi_analysis.get("status") in {
                    "blocked_incomplete_independent_reviews",
                    "blocked_pending_disagreement_adjudication",
                    "multi_review_complete_initial_real_corpus_evidence",
                }
                and multi_analysis.get("proxy_promotion_allowed") is False
            ),
            details={
                "required_for_stage_b_or_stage_c_entry": False,
                "reviewer_a_count": len(reviewer_a_records),
                "reviewer_b_count": len(reviewer_b_records),
                "analysis_status": multi_analysis.get("status"),
            },
        )
    )
    items.append(
        ValidationItem(
            name="temporal_code_stage_b_output_counts",
            ok=len(selected) == int(summary.get("selected_chunks", -1))
            and len(baseline) == int(summary.get("stage_a_random_disjoint_chunks", -1)),
            details={"summary": summary, "selected_lines": len(selected), "baseline_lines": len(baseline)},
        )
    )


def _validate_slm_update_experiment_manifests(items: List[ValidationItem]) -> None:
    manifest_paths = sorted(OUTPUT_DIR.glob(SLM_UPDATE_EXPERIMENT_MANIFEST_GLOB))
    items.append(
        ValidationItem(
            name="slm_update_experiment_manifest_discovery",
            ok=True,
            details={"count": len(manifest_paths), "paths": [str(path) for path in manifest_paths]},
        )
    )
    for manifest_path in manifest_paths:
        manifest = load_json(manifest_path)
        experiment_name = str(manifest.get("experiment_name") or manifest_path.parent.name)
        dataset = str(manifest.get("dataset") or "unknown")
        prefix = f"slm_update_experiment_{experiment_name}_{dataset}"
        framework_scope = manifest.get("framework_scope") if isinstance(manifest.get("framework_scope"), dict) else {}
        target_model = manifest.get("target_model") if isinstance(manifest.get("target_model"), dict) else {}
        budget = manifest.get("budget") if isinstance(manifest.get("budget"), dict) else {}
        inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
        arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
        training_runs = manifest.get("required_training_runs") if isinstance(manifest.get("required_training_runs"), dict) else {}
        required_evaluation = set(_as_str_list(manifest.get("required_evaluation")))
        missing_arms = sorted(SLM_UPDATE_REQUIRED_ARMS - set(map(str, arms.keys())))
        missing_training_runs = sorted(SLM_UPDATE_REQUIRED_TRAINING_RUNS - set(map(str, training_runs.keys())))
        missing_evaluation = sorted(SLM_UPDATE_REQUIRED_EVALUATION - required_evaluation)

        items.append(
            ValidationItem(
                name=f"{prefix}_schema_and_scope",
                ok=manifest.get("schema_version") == "slm-update-experiment-v1"
                and manifest.get("primary_comparison") == "curated_equal_budget_vs_stageA_random_equal_budget"
                and framework_scope.get("stage_a") == "chunk-level hard gate"
                and framework_scope.get("stage_b") == "chunk-level selection"
                and framework_scope.get("stage_c") == "subset-level validation"
                and framework_scope.get("utility_scope") == "Stage C only; never selector objective",
                details={
                    "path": str(manifest_path),
                    "schema_version": manifest.get("schema_version"),
                    "primary_comparison": manifest.get("primary_comparison"),
                    "framework_scope": framework_scope,
                },
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_target_model_preregistered_or_explicitly_unselected",
                ok=target_model.get("status") in {"not_selected", "frozen"}
                and isinstance(target_model.get("selection_rule"), str)
                and bool(str(target_model.get("selection_rule") or "").strip()),
                details={"target_model": target_model},
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_equal_budget_contract",
                ok=budget.get("unit") == "word_count_proxy"
                and isinstance(budget.get("equal_budget_words"), int)
                and int(budget.get("equal_budget_words") or 0) > 0
                and isinstance(budget.get("curated_full_words"), int)
                and int(budget.get("curated_full_words") or 0) > 0,
                details={"budget": budget},
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_input_pools_present",
                ok=Path(str(inputs.get("selected_path") or "")).exists()
                and Path(str(inputs.get("scored_path") or "")).exists()
                and int(inputs.get("selected_records") or 0) > 0
                and int(inputs.get("scored_records") or 0) >= int(inputs.get("selected_records") or 0)
                and int(inputs.get("stage_a_records") or 0) > 0
                and int(inputs.get("stage_a_control_records_excluding_selected") or 0) > 0
                and int(inputs.get("raw_control_records_excluding_selected") or 0) > 0,
                details={"inputs": inputs},
            )
        )

        equal_budget_details: Dict[str, Any] = {}
        equal_budget_ok = not missing_arms
        for arm_name in ("curated_equal_budget", "stageA_random_equal_budget", "raw_random_equal_budget"):
            arm = arms.get(arm_name) if isinstance(arms.get(arm_name), dict) else {}
            arm_path = Path(str(arm.get("path") or ""))
            line_count = _count_lines(arm_path) if arm_path.exists() else None
            equal_budget_details[arm_name] = {
                "path": str(arm.get("path")),
                "path_exists": arm_path.exists(),
                "records": arm.get("records"),
                "line_count": line_count,
                "word_count": arm.get("word_count"),
            }
            equal_budget_ok = equal_budget_ok and arm_path.exists()
            equal_budget_ok = equal_budget_ok and isinstance(arm.get("records"), int) and int(arm.get("records") or 0) > 0
            equal_budget_ok = equal_budget_ok and line_count == int(arm.get("records") or -1)
            equal_budget_ok = equal_budget_ok and isinstance(arm.get("word_count"), int) and int(arm.get("word_count") or 0) > 0

        reference_details: Dict[str, Any] = {}
        reference_ok = not missing_arms
        for arm_name in ("stageA_all_reference", "raw_all_reference"):
            arm = arms.get(arm_name) if isinstance(arms.get(arm_name), dict) else {}
            reference_details[arm_name] = {
                "records": arm.get("records"),
                "word_count": arm.get("word_count"),
                "path": arm.get("path"),
            }
            reference_ok = reference_ok and isinstance(arm.get("records"), int) and int(arm.get("records") or 0) > 0
            reference_ok = reference_ok and isinstance(arm.get("word_count"), int) and int(arm.get("word_count") or 0) > 0

        items.append(
            ValidationItem(
                name=f"{prefix}_required_arms_present",
                ok=not missing_arms,
                details={"missing_arms": missing_arms, "arms": sorted(map(str, arms.keys()))},
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_equal_budget_arm_files",
                ok=bool(equal_budget_ok),
                details=equal_budget_details,
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_reference_arm_summaries",
                ok=bool(reference_ok),
                details=reference_details,
            )
        )
        items.append(
            ValidationItem(
                name=f"{prefix}_training_and_evaluation_contract",
                ok=not missing_training_runs
                and not missing_evaluation
                and isinstance(training_runs.get("min_primary_seeds"), int)
                and int(training_runs.get("min_primary_seeds") or 0) >= 3,
                details={
                    "missing_training_runs": missing_training_runs,
                    "missing_evaluation": missing_evaluation,
                    "min_primary_seeds": training_runs.get("min_primary_seeds"),
                },
            )
        )
        frozen_plan_path = manifest_path.parent / SLM_UPDATE_FROZEN_PLAN_NAME
        if frozen_plan_path.exists():
            plan = load_json(frozen_plan_path)
            plan_scope = plan.get("framework_scope") if isinstance(plan.get("framework_scope"), dict) else {}
            token_budget = plan.get("token_budget") if isinstance(plan.get("token_budget"), dict) else {}
            target_model = plan.get("target_model") if isinstance(plan.get("target_model"), dict) else {}
            tokenizer = plan.get("tokenizer") if isinstance(plan.get("tokenizer"), dict) else {}
            arm_token_counts = plan.get("arm_token_counts") if isinstance(plan.get("arm_token_counts"), dict) else {}
            required_runs = plan.get("required_training_runs") if isinstance(plan.get("required_training_runs"), dict) else {}
            seeds = required_runs.get("seeds") if isinstance(required_runs.get("seeds"), list) else []
            missing_token_arms = sorted(set(SLM_UPDATE_EQUAL_BUDGET_ARMS) - set(map(str, arm_token_counts.keys())))
            token_arm_details: Dict[str, Any] = {}
            token_arms_ok = not missing_token_arms
            for arm_name in SLM_UPDATE_EQUAL_BUDGET_ARMS:
                arm = arm_token_counts.get(arm_name) if isinstance(arm_token_counts.get(arm_name), dict) else {}
                token_arm_details[arm_name] = {
                    "records": arm.get("records"),
                    "nonempty_records": arm.get("nonempty_records"),
                    "word_count": arm.get("word_count"),
                    "token_count": arm.get("token_count"),
                    "max_record_tokens": arm.get("max_record_tokens"),
                    "sha256": arm.get("sha256"),
                }
                token_arms_ok = token_arms_ok and isinstance(arm.get("records"), int) and int(arm.get("records") or 0) > 0
                token_arms_ok = token_arms_ok and arm.get("records") == arm.get("nonempty_records")
                token_arms_ok = token_arms_ok and isinstance(arm.get("token_count"), int) and int(arm.get("token_count") or 0) > 0
                token_arms_ok = token_arms_ok and isinstance(arm.get("sha256"), str) and len(str(arm.get("sha256"))) == 64
            items.append(
                ValidationItem(
                    name=f"{prefix}_frozen_training_plan_scope",
                    ok=plan.get("schema_version") == "slm-update-frozen-plan-v1"
                    and plan.get("primary_comparison") == "curated_equal_budget_vs_stageA_random_equal_budget"
                    and plan_scope.get("utility_scope") == "Stage C only; never selector objective"
                    and target_model.get("selection_status") == "frozen_for_experiment"
                    and isinstance(target_model.get("model_id"), str)
                    and bool(str(target_model.get("model_id") or "").strip())
                    and isinstance(tokenizer.get("vocab_size"), int)
                    and int(tokenizer.get("vocab_size") or 0) > 0,
                    details={
                        "path": str(frozen_plan_path),
                        "target_model": target_model,
                        "tokenizer": tokenizer,
                        "framework_scope": plan_scope,
                    },
                )
            )
            items.append(
                ValidationItem(
                    name=f"{prefix}_frozen_training_plan_token_budget",
                    ok=isinstance(token_budget.get("primary_matched_token_budget"), int)
                    and int(token_budget.get("primary_matched_token_budget") or 0) > 0
                    and isinstance(token_budget.get("all_equal_budget_arms_matched_token_budget"), int)
                    and int(token_budget.get("all_equal_budget_arms_matched_token_budget") or 0) > 0
                    and int(token_budget.get("all_equal_budget_arms_matched_token_budget") or 0)
                    <= min(int((arm_token_counts.get(name) or {}).get("token_count") or 0) for name in SLM_UPDATE_EQUAL_BUDGET_ARMS)
                    and str(token_budget.get("packing_policy") or "").startswith("tokenize_then_pack")
                    and "split_long_records" in str(token_budget.get("overflow_policy") or ""),
                    details={"token_budget": token_budget},
                )
            )
            items.append(
                ValidationItem(
                    name=f"{prefix}_frozen_training_plan_arm_token_counts",
                    ok=bool(token_arms_ok),
                    details={"missing_token_arms": missing_token_arms, "arms": token_arm_details},
                )
            )
            items.append(
                ValidationItem(
                    name=f"{prefix}_frozen_training_plan_training_runs",
                    ok=len(seeds) >= 3
                    and "curated_equal_budget" in _as_str_list(required_runs.get("primary_train_arms"))
                    and "stageA_random_equal_budget" in _as_str_list(required_runs.get("primary_train_arms"))
                    and "raw_random_equal_budget" in _as_str_list(required_runs.get("supporting_train_arms")),
                    details={"required_training_runs": required_runs},
                )
            )


def _replicate_status_by_family(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, Dict[int, bool]]:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return {}
    status_by_family: Dict[str, Dict[int, bool]] = {}
    for preset, run in runs.items():
        if not isinstance(run, dict) or not run.get("exists") or not run.get("compatible"):
            continue
        match = REPLICATE_PRESET_RE.match(str(preset))
        if not match:
            continue
        family = match.group("family")
        replicate = int(match.group("replicate"))
        status_by_family.setdefault(family, {})[replicate] = bool(
            run.get("probe_valid") and run.get("selected_gt_random")
        )
    return status_by_family


def _replicated_valid_family_replicates(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, List[int]]:
    status_by_family = _replicate_status_by_family(power_sweep, dataset)
    return {
        family: sorted(status_by_replicate)
        for family, status_by_replicate in sorted(status_by_family.items())
        if len(status_by_replicate) >= 2 and all(status_by_replicate.values())
    }


def _replicated_valid_families(power_sweep: Dict[str, Any], dataset: str) -> List[str]:
    return sorted(_replicated_valid_family_replicates(power_sweep, dataset).keys())


def _best_replicated_preset(power_sweep: Dict[str, Any], dataset: str) -> str | None:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return None
    replicated = _replicated_valid_family_replicates(power_sweep, dataset)
    candidates = [
        f"{family}_b{replicate}"
        for family, replicates in replicated.items()
        for replicate in replicates
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (
            float((runs.get(name) or {}).get("selected_minus_random") or 0.0),
            str(name),
        ),
    )


def _best_replicated_family(power_sweep: Dict[str, Any], dataset: str) -> str | None:
    preset = _best_replicated_preset(power_sweep, dataset)
    match = REPLICATE_PRESET_RE.match(str(preset or ""))
    return match.group("family") if match else None


def _power_sweep_decision(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    decision = payload.get("decision") if isinstance(payload, dict) else None
    return decision if isinstance(decision, dict) else {}


def _infer_report_profile(run_summary: Dict[str, Any]) -> str:
    profiles = run_summary.get("profiles") if isinstance(run_summary, dict) else None
    if not isinstance(profiles, dict):
        return "canonical"
    names = [
        str(name)
        for name, payload in profiles.items()
        if not str(name).startswith("_") and isinstance(payload, dict)
    ]
    if not names:
        return "canonical"
    if len(names) == 1:
        return names[0]
    for preferred in ("paper_release_certification", "core_proxy_length_recurrence_guard", "canonical"):
        if preferred in names:
            return preferred
    return names[0]


def _validate_scored_file(path: Path) -> List[ValidationItem]:
    failures: List[ValidationItem] = []
    if not path.exists():
        return [ValidationItem(name="scored_exists", ok=False, details={"path": str(path)})]
    for idx, raw in enumerate(iter_nonempty_lines_resilient(path), start=1):
        record = json.loads(raw)
        if record.get("schema_version") != SCHEMA_VERSION:
            failures.append(ValidationItem(name="scored_schema", ok=False, details={"path": str(path), "line": idx}))
            break
        core_metrics = record.get("core_metrics") or {}
        diagnostic_metrics = record.get("diagnostic_metrics") or {}
        if set(core_metrics.keys()) != set(CORE_SELECTION_METRICS):
            failures.append(ValidationItem(name="scored_core_metric_keys", ok=False, details={"path": str(path), "line": idx, "keys": sorted(core_metrics.keys())}))
            break
        if set(diagnostic_metrics.keys()) != set(DIAGNOSTIC_METRICS):
            failures.append(ValidationItem(name="scored_diagnostic_metric_keys", ok=False, details={"path": str(path), "line": idx, "keys": sorted(diagnostic_metrics.keys())}))
            break
        validity_details = (diagnostic_metrics.get("structural_validity_score") or {}).get("details") or {}
        if validity_details.get("decision_scope") != "structural_usability_only":
            failures.append(
                ValidationItem(
                    name="scored_validity_contract_fields",
                    ok=False,
                    details={
                        "path": str(path),
                        "line": idx,
                        "decision_scope": validity_details.get("decision_scope"),
                    },
                )
            )
            break
        quality_details = (core_metrics.get("reference_quality_score") or {}).get("details") or {}
        if quality_details.get("quality_calibration_policy") != "style_length_normalized_quality_v2":
            failures.append(
                ValidationItem(
                    name="scored_quality_calibration_contract_fields",
                    ok=False,
                    details={
                        "path": str(path),
                        "line": idx,
                        "quality_calibration_policy": quality_details.get("quality_calibration_policy"),
                    },
                )
            )
            break
        if "style_length_normalized_quality" not in quality_details or "quality_evidence_score" not in quality_details:
            failures.append(
                ValidationItem(
                    name="scored_quality_calibration_v2_fields",
                    ok=False,
                    details={
                        "path": str(path),
                        "line": idx,
                        "quality_details": quality_details,
                    },
                )
            )
            break
        redundancy_details = (core_metrics.get("shingle_near_duplicate_risk_score") or {}).get("details") or {}
        if redundancy_details.get("redundancy_policy") != "harmful_redundancy_minus_useful_recurrence_v1":
            failures.append(
                ValidationItem(
                    name="scored_redundancy_policy_contract_fields",
                    ok=False,
                    details={
                        "path": str(path),
                        "line": idx,
                        "redundancy_policy": redundancy_details.get("redundancy_policy"),
                    },
                )
            )
            break
        if "selection" in record:
            failures.append(ValidationItem(name="scored_should_be_threshold_free", ok=False, details={"path": str(path), "line": idx}))
            break
    return failures or [ValidationItem(name="scored_schema", ok=True, details={"path": str(path)})]


def _validate_profile_semantics(run_manifest: Dict[str, Any]) -> List[ValidationItem]:
    items: List[ValidationItem] = []
    profiles = run_manifest.get("profiles") or {}
    available_profiles = [name for name, payload in profiles.items() if isinstance(payload, dict)]
    items.append(
        ValidationItem(
            name="profile_semantics_profile_count",
            ok=len(available_profiles) >= 1,
            details={"available_profiles": available_profiles},
        )
    )
    if len(available_profiles) <= 1:
        return items
    ordered_profiles = [name for name in LEGACY_VARIANT_PROFILE_ORDER if name in profiles]
    if len(ordered_profiles) < 2:
        items.append(
            ValidationItem(
                name="profile_semantics_variant_family_optional",
                ok=True,
                details={"available_profiles": available_profiles},
            )
        )
        return items

    thresholds = [float((profiles[name] or {}).get("selection_threshold") or 0.0) for name in ordered_profiles]
    items.append(
        ValidationItem(
            name="profile_threshold_order",
            ok=all(thresholds[i] >= thresholds[i + 1] for i in range(len(thresholds) - 1)),
            details={"profiles": ordered_profiles, "selection_thresholds": thresholds},
        )
    )

    floor_metrics = sorted(
        set.intersection(
            *(set((profiles[name] or {}).get("metric_floors", {}).keys()) for name in ordered_profiles)
        )
        if ordered_profiles
        else set()
    )
    for metric_name in floor_metrics:
        values = [float(profiles[name]["metric_floors"][metric_name]) for name in ordered_profiles]
        items.append(
            ValidationItem(
                name=f"profile_floor_order_{metric_name}",
                ok=all(values[i] >= values[i + 1] for i in range(len(values) - 1)),
                details={"profiles": ordered_profiles, "values": values},
            )
        )

    ceiling_metrics = sorted(
        set.intersection(
            *(set((profiles[name] or {}).get("metric_ceilings", {}).keys()) for name in ordered_profiles)
        )
        if ordered_profiles
        else set()
    )
    for metric_name in ceiling_metrics:
        values = [float(profiles[name]["metric_ceilings"][metric_name]) for name in ordered_profiles]
        items.append(
            ValidationItem(
                name=f"profile_ceiling_order_{metric_name}",
                ok=all(values[i] <= values[i + 1] for i in range(len(values) - 1)),
                details={"profiles": ordered_profiles, "values": values},
            )
        )

    dataset_keys = sorted(
        set.intersection(
            *(set((profiles[name] or {}).get("datasets", {}).keys()) for name in ordered_profiles)
        )
        if ordered_profiles
        else set()
    )
    for dataset in dataset_keys:
        selected_counts = [int(profiles[name]["datasets"][dataset]["selected_records"]) for name in ordered_profiles]
        items.append(
            ValidationItem(
                name=f"profile_selected_order_{dataset}",
                ok=all(selected_counts[i] <= selected_counts[i + 1] for i in range(len(selected_counts) - 1)),
                details={"profiles": ordered_profiles, "selected_records": selected_counts},
            )
        )
        coverage_scores = [float(profiles[name]["datasets"][dataset]["subset_coverage_retention_score"]) for name in ordered_profiles]
        items.append(
            ValidationItem(
                name=f"profile_coverage_order_{dataset}",
                ok=all(coverage_scores[i] <= coverage_scores[i + 1] for i in range(len(coverage_scores) - 1)),
                details={"profiles": ordered_profiles, "subset_coverage_retention_score": coverage_scores},
            )
        )

    return items


def _validate_metric_spec() -> List[ValidationItem]:
    if not METRIC_SPEC_PATH.exists():
        return [ValidationItem(name="metric_spec_exists", ok=False, details={"path": str(METRIC_SPEC_PATH)})]

    spec = load_json(METRIC_SPEC_PATH)
    items: List[ValidationItem] = [
        ValidationItem(
            name="metric_spec_schema",
            ok=spec.get("schema_version") == METRIC_SPEC_SCHEMA_VERSION,
            details={"schema_version": spec.get("schema_version")},
        )
    ]

    metrics = spec.get("metrics") or {}
    metric_keys = set(metrics.keys())
    items.append(
        ValidationItem(
            name="metric_spec_metric_keys",
            ok=metric_keys == set(ALL_METRICS),
            details={"metric_keys": sorted(metric_keys), "expected": sorted(ALL_METRICS)},
        )
    )

    paper_registry = spec.get("paper_registry") or {}
    items.append(
        ValidationItem(
            name="metric_spec_paper_registry",
            ok=bool(paper_registry),
            details={"paper_count": len(paper_registry)},
        )
    )

    suite = spec.get("property_benchmark_suite") or {}
    items.append(
        ValidationItem(
            name="metric_spec_property_suite",
            ok=bool(suite.get("buckets")) and bool(suite.get("assertions")),
            details={
                "bucket_count": len(suite.get("buckets") or {}),
                "assertion_count": len(suite.get("assertions") or []),
            },
        )
    )

    contract_violations: List[Dict[str, Any]] = []

    for metric_name in sorted(metric_keys):
        meta = metrics.get(metric_name) or {}
        required_fields = (
            "role",
            "status",
            "claim",
            "paper_ids",
            "formal_definition",
            "implementation",
            "expected_behavior",
            "failure_modes",
            "acceptance_tests",
        )
        missing = [field for field in required_fields if not meta.get(field)]
        items.append(
            ValidationItem(
                name=f"metric_spec_fields_{metric_name}",
                ok=not missing,
                details={"missing": missing},
            )
        )

        items.append(
            ValidationItem(
                name=f"metric_spec_role_{metric_name}",
                ok=str(meta.get("role") or "") in METRIC_ROLES,
                details={"role": meta.get("role")},
            )
        )
        items.append(
            ValidationItem(
                name=f"metric_spec_status_{metric_name}",
                ok=str(meta.get("status") or "") in METRIC_STATUSES,
                details={"status": meta.get("status")},
            )
        )

        implementation = meta.get("implementation") or {}
        impl_path = implementation.get("path")
        impl_file = (Path(__file__).resolve().parent / str(impl_path)).resolve() if impl_path else None
        items.append(
            ValidationItem(
                name=f"metric_spec_implementation_{metric_name}",
                ok=bool(impl_file and impl_file.exists()),
                details={"path": str(impl_file) if impl_file else None, "entrypoint": implementation.get("entrypoint")},
            )
        )

        unknown_papers = [paper_id for paper_id in meta.get("paper_ids", []) if paper_id not in paper_registry]
        items.append(
            ValidationItem(
                name=f"metric_spec_papers_{metric_name}",
                ok=not unknown_papers,
                details={"unknown_paper_ids": unknown_papers},
            )
        )

        contract = meta.get("orthogonality_contract") or {}
        allowed = contract.get("allowed_signals") or []
        prohibited = contract.get("prohibited_signals") or []
        axis = str(contract.get("axis") or "")
        items.append(
            ValidationItem(
                name=f"metric_spec_contract_fields_{metric_name}",
                ok=bool(axis) and isinstance(allowed, list) and isinstance(prohibited, list),
                details={"axis": axis, "allowed_type": type(allowed).__name__, "prohibited_type": type(prohibited).__name__},
            )
        )
        if metric_name in THEORY_AXIS_EXPECTED:
            expected_axis = THEORY_AXIS_EXPECTED[metric_name]
            ok_axis = axis == expected_axis
            items.append(
                ValidationItem(
                    name=f"metric_spec_contract_axis_{metric_name}",
                    ok=ok_axis,
                    details={"axis": axis, "expected_axis": expected_axis},
                )
            )
            if not ok_axis:
                contract_violations.append({"metric": metric_name, "axis": axis, "expected": expected_axis})
        if metric_name in THEORY_AXIS_EXPECTED and (not allowed or not prohibited):
            contract_violations.append({"metric": metric_name, "reason": "missing_allowed_or_prohibited_signals"})

    items.append(
        ValidationItem(
            name="theory_contract_violations",
            ok=not contract_violations,
            details={"violations": contract_violations},
        )
    )
    evidence_audit = load_json(METRIC_EVIDENCE_AUDIT_PATH) if METRIC_EVIDENCE_AUDIT_PATH.exists() else {}
    evidence_components = evidence_audit.get("components") if isinstance(evidence_audit.get("components"), dict) else {}
    project_specific = [
        row
        for row in evidence_components.values()
        if "project-specific" in str((row or {}).get("parameter_origin") or "")
    ]
    optional_review = evidence_audit.get("human_or_llm_review") if isinstance(evidence_audit.get("human_or_llm_review"), dict) else {}
    items.append(
        ValidationItem(
            name="metric_evidence_audit_claim_boundary",
            ok=evidence_audit.get("schema_version") == "metric-evidence-audit-v1"
            and bool(project_specific)
            and all((row or {}).get("evidence_class") == "project_hypothesis_frozen" for row in project_specific)
            and optional_review.get("required_for_stage_b_approval") is False
            and optional_review.get("required_for_stage_c_entry") is False
            and optional_review.get("may_tune_or_promote_selector") is False
            and (evidence_audit.get("known_citation_gap") or {}).get("status") == "incomplete",
            details={
                "path": str(METRIC_EVIDENCE_AUDIT_PATH),
                "component_count": len(evidence_components),
                "project_specific_parameter_count": len(project_specific),
                "known_citation_gap": (evidence_audit.get("known_citation_gap") or {}).get("status"),
            },
        )
    )

    return items


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return 0.0
    a_rank = np.argsort(np.argsort(a)).astype(np.float64)
    b_rank = np.argsort(np.argsort(b)).astype(np.float64)
    a_rank -= float(a_rank.mean())
    b_rank -= float(b_rank.mean())
    denom = float(np.sqrt(np.sum(a_rank * a_rank) * np.sum(b_rank * b_rank)))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(a_rank * b_rank) / denom)


def _record_metric_payload(record: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    core = record.get("core_metrics") or {}
    diagnostic = record.get("diagnostic_metrics") or {}
    payload = core.get(metric_name) or diagnostic.get(metric_name)
    if not isinstance(payload, dict):
        raise KeyError(metric_name)
    return payload


def _orthogonality_items(scoring_manifest: Dict[str, Any]) -> List[ValidationItem]:
    items: List[ValidationItem] = []
    metric_pairs = (
        ("structural_validity_score", "reference_quality_score"),
        ("structural_validity_score", "shingle_near_duplicate_risk_score"),
        ("reference_quality_score", "shingle_near_duplicate_risk_score"),
    )
    for dataset, meta in (scoring_manifest.get("datasets") or {}).items():
        path = Path(str(meta.get("path") or ""))
        if not path.exists():
            items.append(
                ValidationItem(
                    name=f"orthogonality_scored_exists_{dataset}",
                    ok=False,
                    details={"path": str(path)},
                )
            )
            continue
        values: Dict[str, List[float]] = {name: [] for pair in metric_pairs for name in pair}
        for idx, record in enumerate(iter_jsonl_records_resilient(path)):
            try:
                for metric_name in values:
                    values[metric_name].append(float(_record_metric_payload(record, metric_name)["score"]))
            except KeyError:
                continue
            if idx + 1 >= ORTHOGONALITY_SAMPLE_LIMIT:
                break

        for left, right in metric_pairs:
            if not values[left] or not values[right]:
                items.append(
                    ValidationItem(
                        name=f"orthogonality_{dataset}_{left}_{right}",
                        ok=False,
                        details={"reason": "insufficient_values"},
                    )
                )
                continue
            rho = _spearman(np.asarray(values[left], dtype=np.float64), np.asarray(values[right], dtype=np.float64))
            items.append(
                ValidationItem(
                    name=f"orthogonality_{dataset}_{left}_{right}",
                    ok=abs(rho) <= ORTHOGONALITY_MAX_ABS_SPEARMAN,
                    details={
                        "spearman": round(float(rho), 6),
                        "max_abs_spearman": ORTHOGONALITY_MAX_ABS_SPEARMAN,
                        "sampled_rows": len(values[left]),
                    },
                )
            )
    return items


def _validate_utility_axis_no_metric_leakage() -> ValidationItem:
    path = Path(__file__).resolve().parent / "utility" / "lm_probe.py"
    if not path.exists():
        return ValidationItem(
            name="orthogonality_utility_probe_file_exists",
            ok=False,
            details={"path": str(path)},
        )
    body = path.read_text(encoding="utf-8", errors="replace")
    forbidden_patterns = (
        r"\breference_quality_score\b",
        r"\bshingle_near_duplicate_risk_score\b",
        r"\bexact_duplicate_indicator\b",
        r"\bpredictive_utility_proxy\b",
        r"\butility_feature_vector\b",
    )
    hits = [pat for pat in forbidden_patterns if re.search(pat, body)]
    return ValidationItem(
        name="orthogonality_utility_axis_leakage",
        ok=not hits,
        details={"forbidden_hits": hits, "path": str(path)},
    )


def _validate_selector_no_utility_surrogate() -> ValidationItem:
    path = Path(__file__).resolve().parent / "policy" / "subsets.py"
    body = path.read_text(encoding="utf-8", errors="replace")
    forbidden_snippets = (
        'weights["utility_surrogate"]',
        'components["utility_surrogate"]',
        'weights["diagnostic_predictive_utility"]',
        'components["diagnostic_predictive_utility"]',
    )
    hits = [snippet for snippet in forbidden_snippets if snippet in body]
    return ValidationItem(
        name="theory_contract_selector_no_utility_surrogate",
        ok=not hits,
        details={"forbidden_hits": hits, "path": str(path)},
    )


def _validate_profile_configs_no_utility_surrogate() -> ValidationItem:
    config_dir = Path(__file__).resolve().parent / "configs"
    paths = sorted(config_dir.glob("curation_profiles*.json"))
    offenders: List[Dict[str, Any]] = []

    for path in paths:
        try:
            payload = load_json(path)
        except Exception as exc:
            offenders.append({"path": str(path), "reason": f"unreadable: {exc}"})
            continue
        for profile_name, profile in (payload.get("profiles") or {}).items():
            stage_b = (profile or {}).get("stage_b_rank") or {}
            weights = stage_b.get("weights") or {}
            if "utility_surrogate" in weights:
                offenders.append(
                    {
                        "path": str(path),
                        "profile": profile_name,
                        "location": "stage_b_rank.weights.utility_surrogate",
                    }
                )
            unexpected = sorted(set(weights.keys()) - {"quality", "redundancy"})
            if unexpected:
                offenders.append(
                    {
                        "path": str(path),
                        "profile": profile_name,
                        "location": "stage_b_rank.weights",
                        "unexpected_weight_keys": unexpected,
                    }
                )

    return ValidationItem(
        name="theory_contract_profile_configs_no_utility_surrogate",
        ok=not offenders,
        details={"checked_files": [str(path) for path in paths], "offenders": offenders[:20]},
    )


def validate_outputs(scope: str = "full") -> List[ValidationItem]:
    historical_evidence_enabled = includes_historical_evidence(scope)
    items: List[ValidationItem] = []
    if STAGE0_PROCESSING_REPORT_PATH.exists():
        stage0_processing = load_json(STAGE0_PROCESSING_REPORT_PATH)
        stage0_summary = stage0_processing.get("summary") or {}
        stage0_outputs = stage0_processing.get("outputs") or {}
        release_path = Path(str(stage0_outputs.get("release_candidates") or ""))
        quarantine_path = Path(str(stage0_outputs.get("quarantined_candidates") or ""))
        items.append(
            ValidationItem(
                name="stage0_processing_report",
                ok=stage0_processing.get("schema_version") == "stage0-processing-report-v1"
                and stage0_processing.get("contract") == "candidate-corpus-record-v1"
                and int(stage0_summary.get("input_records") or 0)
                == int(stage0_summary.get("release_candidate_records") or 0)
                + int(stage0_summary.get("quarantined_records") or 0)
                and release_path.exists()
                and quarantine_path.exists()
                and isinstance(stage0_summary.get("quarantine_reason_counts"), dict),
                details={"path": str(STAGE0_PROCESSING_REPORT_PATH), "summary": stage0_summary},
            )
        )
    if STAGE0_CONTRACT_VALIDATION_PATH.exists():
        stage0_report = load_json(STAGE0_CONTRACT_VALIDATION_PATH)
        stage0_summary = stage0_report.get("summary") or {}
        stage0_records = stage0_report.get("records") or {}
        items.append(
            ValidationItem(
                name="stage0_contract_validation",
                ok=stage0_report.get("schema_version") == "stage0-contract-validation-v1"
                and stage0_report.get("candidate_record_schema_version") == "candidate-corpus-record-v1"
                and isinstance(stage0_records, dict)
                and int(stage0_summary.get("record_count") or 0) > 0
                and stage0_summary.get("contract_valid_count") == stage0_summary.get("record_count")
                and all(
                    isinstance(payload, dict)
                    and isinstance(payload.get("eligible"), bool)
                    and isinstance(payload.get("blockers"), list)
                    and isinstance(payload.get("validation_errors"), list)
                    for payload in stage0_records.values()
                ),
                details={"path": str(STAGE0_CONTRACT_VALIDATION_PATH), "summary": stage0_summary},
            )
        )
    if OPENWEBTEXT2_SLICE_DIAGNOSTIC_PATH.exists():
        openweb_diagnostic = load_json(OPENWEBTEXT2_SLICE_DIAGNOSTIC_PATH)
        openweb_slices = openweb_diagnostic.get("slices") or {}
        required_slices = {"selected", "stage_a_usable_not_selected", "stage_a_rejected"}
        items.append(
            ValidationItem(
                name="openwebtext2_slice_diagnostic",
                ok=openweb_diagnostic.get("schema_version") == "openwebtext2-slice-diagnostic-v1"
                and openweb_diagnostic.get("dataset") == "openwebtext2_subset"
                and openweb_diagnostic.get("scope") == "diagnostic only"
                and openweb_diagnostic.get("selector_action") == "hold"
                and required_slices.issubset(set(openweb_slices))
                and all(
                    int((openweb_slices.get(name) or {}).get("records") or 0) > 0
                    and isinstance((openweb_slices.get(name) or {}).get("feature_means"), dict)
                    for name in required_slices
                )
                and isinstance(openweb_diagnostic.get("hypotheses"), list),
                details={
                    "path": str(OPENWEBTEXT2_SLICE_DIAGNOSTIC_PATH),
                    "slice_records": {
                        name: (openweb_slices.get(name) or {}).get("records")
                        for name in sorted(required_slices)
                    },
                },
            )
        )
    items.extend(_validate_metric_spec())
    missing_critical: List[ValidationItem] = []
    if not RUN_MANIFEST_PATH.exists():
        missing_critical.append(ValidationItem(name="run_manifest_exists", ok=False, details={"path": str(RUN_MANIFEST_PATH)}))
    if not RUN_SUMMARY_PATH.exists():
        missing_critical.append(ValidationItem(name="run_summary_exists", ok=False, details={"path": str(RUN_SUMMARY_PATH)}))
    if not SCORING_MANIFEST_PATH.exists():
        missing_critical.append(ValidationItem(name="scoring_manifest_exists", ok=False, details={"path": str(SCORING_MANIFEST_PATH)}))
    if not UTILITY_PROBE_RESULTS_PATH.exists():
        missing_critical.append(ValidationItem(name="utility_probe_results_exists", ok=False, details={"path": str(UTILITY_PROBE_RESULTS_PATH)}))
    if missing_critical:
        return items + missing_critical
    run_summary_for_autobuild = load_json(RUN_SUMMARY_PATH)
    report_profile = _infer_report_profile(run_summary_for_autobuild)
    items.append(
        ValidationItem(
            name="report_autobuild_profile_inferred",
            ok=bool(report_profile),
            details={"profile": report_profile},
        )
    )

    if not DASHBOARD_PATH.exists():
        try:
            build_dashboard()
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="dashboard_exists",
                    ok=False,
                    details={"path": str(DASHBOARD_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="dashboard_autobuilt",
                    ok=True,
                    details={"path": str(DASHBOARD_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="dashboard_exists",
            ok=DASHBOARD_PATH.exists(),
            details={"path": str(DASHBOARD_PATH)},
        )
    )
    if SELECTOR_BASELINE_AUDIT_PATH.exists() and not UTILITY_TRANSFER_GAP_REPORT_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "21_build_utility_transfer_gap_report.py"
            spec = importlib.util.spec_from_file_location("utility_transfer_gap_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(run_summary_for_autobuild, report_profile)
            save_json(UTILITY_TRANSFER_GAP_REPORT_PATH, report)
            module.write_markdown(report, UTILITY_TRANSFER_GAP_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="utility_transfer_gap_report_autobuilt",
                    ok=False,
                    details={"path": str(UTILITY_TRANSFER_GAP_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="utility_transfer_gap_report_autobuilt",
                    ok=True,
                    details={"path": str(UTILITY_TRANSFER_GAP_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="utility_transfer_gap_report_exists_or_not_applicable",
            ok=(not SELECTOR_BASELINE_AUDIT_PATH.exists())
            or (UTILITY_TRANSFER_GAP_REPORT_PATH.exists() and UTILITY_TRANSFER_GAP_REPORT_MD_PATH.exists()),
            details={
                "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
                "json": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
                "markdown": str(UTILITY_TRANSFER_GAP_REPORT_MD_PATH),
            },
        )
    )

    if (
        SELECTOR_BASELINE_AUDIT_PATH.exists()
        and UTILITY_TRANSFER_GAP_REPORT_PATH.exists()
        and not CORE_PROXY_ALIGNMENT_REPORT_PATH.exists()
    ):
        try:
            script_path = Path(__file__).resolve().parent / "23_build_core_proxy_alignment_report.py"
            spec = importlib.util.spec_from_file_location("core_proxy_alignment_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                selector_audit=load_json(SELECTOR_BASELINE_AUDIT_PATH),
                transfer_report=load_json(UTILITY_TRANSFER_GAP_REPORT_PATH),
                power_sweep=load_json(UTILITY_POWER_SWEEP_REPORT_PATH) if UTILITY_POWER_SWEEP_REPORT_PATH.exists() else {},
            )
            save_json(CORE_PROXY_ALIGNMENT_REPORT_PATH, report)
            module.write_markdown(report, CORE_PROXY_ALIGNMENT_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="core_proxy_alignment_report_autobuilt",
                    ok=False,
                    details={"path": str(CORE_PROXY_ALIGNMENT_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="core_proxy_alignment_report_autobuilt",
                    ok=True,
                    details={"path": str(CORE_PROXY_ALIGNMENT_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="core_proxy_alignment_report_exists_or_not_applicable",
            ok=(not SELECTOR_BASELINE_AUDIT_PATH.exists())
            or (not UTILITY_TRANSFER_GAP_REPORT_PATH.exists())
            or (CORE_PROXY_ALIGNMENT_REPORT_PATH.exists() and CORE_PROXY_ALIGNMENT_REPORT_MD_PATH.exists()),
            details={
                "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
                "utility_transfer_gap_report": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
                "json": str(CORE_PROXY_ALIGNMENT_REPORT_PATH),
                "markdown": str(CORE_PROXY_ALIGNMENT_REPORT_MD_PATH),
            },
        )
    )

    if (
        SELECTOR_BASELINE_AUDIT_PATH.exists()
        and UTILITY_TRANSFER_GAP_REPORT_PATH.exists()
        and CORE_PROXY_ALIGNMENT_REPORT_PATH.exists()
        and not CORE_PROXY_CALIBRATION_REPORT_PATH.exists()
    ):
        try:
            script_path = Path(__file__).resolve().parent / "24_build_core_proxy_calibration_report.py"
            spec = importlib.util.spec_from_file_location("core_proxy_calibration_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                selector_audit=load_json(SELECTOR_BASELINE_AUDIT_PATH),
                transfer_report=load_json(UTILITY_TRANSFER_GAP_REPORT_PATH),
                alignment_report=load_json(CORE_PROXY_ALIGNMENT_REPORT_PATH),
                policy_ablation=load_json(POLICY_ABLATION_AUDIT_PATH) if POLICY_ABLATION_AUDIT_PATH.exists() else {},
            )
            save_json(CORE_PROXY_CALIBRATION_REPORT_PATH, report)
            module.write_markdown(report, CORE_PROXY_CALIBRATION_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="core_proxy_calibration_report_autobuilt",
                    ok=False,
                    details={"path": str(CORE_PROXY_CALIBRATION_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="core_proxy_calibration_report_autobuilt",
                    ok=True,
                    details={"path": str(CORE_PROXY_CALIBRATION_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="core_proxy_calibration_report_exists_or_not_applicable",
            ok=(not SELECTOR_BASELINE_AUDIT_PATH.exists())
            or (not UTILITY_TRANSFER_GAP_REPORT_PATH.exists())
            or (not CORE_PROXY_ALIGNMENT_REPORT_PATH.exists())
            or (CORE_PROXY_CALIBRATION_REPORT_PATH.exists() and CORE_PROXY_CALIBRATION_REPORT_MD_PATH.exists()),
            details={
                "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
                "utility_transfer_gap_report": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
                "core_proxy_alignment_report": str(CORE_PROXY_ALIGNMENT_REPORT_PATH),
                "json": str(CORE_PROXY_CALIBRATION_REPORT_PATH),
                "markdown": str(CORE_PROXY_CALIBRATION_REPORT_MD_PATH),
            },
        )
    )

    if not CURATION_READINESS_REPORT_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "20_build_curation_readiness_report.py"
            spec = importlib.util.spec_from_file_location("curation_readiness_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(run_summary_for_autobuild, report_profile)
            save_json(CURATION_READINESS_REPORT_PATH, report)
            module.write_markdown(report, CURATION_READINESS_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="curation_readiness_report_autobuilt",
                    ok=False,
                    details={"path": str(CURATION_READINESS_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="curation_readiness_report_autobuilt",
                    ok=True,
                    details={"path": str(CURATION_READINESS_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="curation_readiness_report_exists",
            ok=CURATION_READINESS_REPORT_PATH.exists() and CURATION_READINESS_REPORT_MD_PATH.exists(),
            details={"json": str(CURATION_READINESS_REPORT_PATH), "markdown": str(CURATION_READINESS_REPORT_MD_PATH)},
        )
    )
    if CURATION_READINESS_REPORT_PATH.exists() and not STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "25_build_stage_c_protocol_decision_report.py"
            spec = importlib.util.spec_from_file_location("stage_c_protocol_decision_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                load_json(CURATION_READINESS_REPORT_PATH),
                load_json(UTILITY_TRANSFER_GAP_REPORT_PATH) if UTILITY_TRANSFER_GAP_REPORT_PATH.exists() else {},
                load_json(UTILITY_POWER_SWEEP_REPORT_PATH)
                if UTILITY_POWER_SWEEP_REPORT_PATH.exists()
                else {},
                module._load_anti_memorization_reports(ANTI_MEMORIZATION_PROBE_REPORT_PATH),
            )
            save_json(STAGE_C_PROTOCOL_DECISION_REPORT_PATH, report)
            module.write_markdown(report, STAGE_C_PROTOCOL_DECISION_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="stage_c_protocol_decision_report_autobuilt",
                    ok=False,
                    details={"path": str(STAGE_C_PROTOCOL_DECISION_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="stage_c_protocol_decision_report_autobuilt",
                    ok=True,
                    details={"path": str(STAGE_C_PROTOCOL_DECISION_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="stage_c_protocol_decision_report_exists",
            ok=STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists() and STAGE_C_PROTOCOL_DECISION_REPORT_MD_PATH.exists(),
            details={
                "json": str(STAGE_C_PROTOCOL_DECISION_REPORT_PATH),
                "markdown": str(STAGE_C_PROTOCOL_DECISION_REPORT_MD_PATH),
            },
        )
    )
    if STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists() and not STRICT_BASELINE_CONTROL_REPORT_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "26_build_strict_baseline_control_report.py"
            spec = importlib.util.spec_from_file_location("strict_baseline_control_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                load_json(CURATION_READINESS_REPORT_PATH) if CURATION_READINESS_REPORT_PATH.exists() else {},
                load_json(UTILITY_TRANSFER_GAP_REPORT_PATH) if UTILITY_TRANSFER_GAP_REPORT_PATH.exists() else {},
                load_json(STAGE_C_PROTOCOL_DECISION_REPORT_PATH),
            )
            save_json(STRICT_BASELINE_CONTROL_REPORT_PATH, report)
            module.write_markdown(report, STRICT_BASELINE_CONTROL_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="strict_baseline_control_report_autobuilt",
                    ok=False,
                    details={"path": str(STRICT_BASELINE_CONTROL_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="strict_baseline_control_report_autobuilt",
                    ok=True,
                    details={"path": str(STRICT_BASELINE_CONTROL_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="strict_baseline_control_report_exists",
            ok=STRICT_BASELINE_CONTROL_REPORT_PATH.exists() and STRICT_BASELINE_CONTROL_REPORT_MD_PATH.exists(),
            details={
                "json": str(STRICT_BASELINE_CONTROL_REPORT_PATH),
                "markdown": str(STRICT_BASELINE_CONTROL_REPORT_MD_PATH),
            },
        )
    )
    if STRICT_BASELINE_CONTROL_REPORT_PATH.exists() and not CURATION_DECISION_REPORT_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "27_build_curation_decision_report.py"
            spec = importlib.util.spec_from_file_location("curation_decision_report_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                load_json(RUN_SUMMARY_PATH) if RUN_SUMMARY_PATH.exists() else {},
                load_json(CURATION_READINESS_REPORT_PATH) if CURATION_READINESS_REPORT_PATH.exists() else {},
                load_json(STAGE_C_PROTOCOL_DECISION_REPORT_PATH) if STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists() else {},
                load_json(STRICT_BASELINE_CONTROL_REPORT_PATH),
                load_json(SELECTOR_BASELINE_AUDIT_PATH) if SELECTOR_BASELINE_AUDIT_PATH.exists() else {},
            )
            save_json(CURATION_DECISION_REPORT_PATH, report)
            module.write_markdown(report, CURATION_DECISION_REPORT_MD_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="curation_decision_report_autobuilt",
                    ok=False,
                    details={"path": str(CURATION_DECISION_REPORT_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="curation_decision_report_autobuilt",
                    ok=True,
                    details={"path": str(CURATION_DECISION_REPORT_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="curation_decision_report_exists",
            ok=CURATION_DECISION_REPORT_PATH.exists() and CURATION_DECISION_REPORT_MD_PATH.exists(),
            details={
                "json": str(CURATION_DECISION_REPORT_PATH),
                "markdown": str(CURATION_DECISION_REPORT_MD_PATH),
            },
        )
    )
    if CURATION_DECISION_REPORT_PATH.exists() and not PAPER_EVIDENCE_TABLE_PATH.exists():
        try:
            script_path = Path(__file__).resolve().parent / "28_build_paper_evidence_table.py"
            spec = importlib.util.spec_from_file_location("paper_evidence_table_builder", script_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot import {script_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.build_report(
                load_json(RUN_SUMMARY_PATH) if RUN_SUMMARY_PATH.exists() else {},
                load_json(SELECTOR_BASELINE_AUDIT_PATH) if SELECTOR_BASELINE_AUDIT_PATH.exists() else {},
                load_json(CURATION_READINESS_REPORT_PATH) if CURATION_READINESS_REPORT_PATH.exists() else {},
                load_json(STAGE_C_PROTOCOL_DECISION_REPORT_PATH) if STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists() else {},
                load_json(STRICT_BASELINE_CONTROL_REPORT_PATH) if STRICT_BASELINE_CONTROL_REPORT_PATH.exists() else {},
                load_json(CURATION_DECISION_REPORT_PATH),
            )
            save_json(PAPER_EVIDENCE_TABLE_PATH, report)
            module.write_markdown(report, PAPER_EVIDENCE_TABLE_MD_PATH)
            module.write_csv(report, PAPER_EVIDENCE_TABLE_CSV_PATH)
        except Exception as exc:
            items.append(
                ValidationItem(
                    name="paper_evidence_table_autobuilt",
                    ok=False,
                    details={"path": str(PAPER_EVIDENCE_TABLE_PATH), "autobuild_error": str(exc)},
                )
            )
        else:
            items.append(
                ValidationItem(
                    name="paper_evidence_table_autobuilt",
                    ok=True,
                    details={"path": str(PAPER_EVIDENCE_TABLE_PATH)},
                )
            )
    items.append(
        ValidationItem(
            name="paper_evidence_table_exists",
            ok=PAPER_EVIDENCE_TABLE_PATH.exists()
            and PAPER_EVIDENCE_TABLE_MD_PATH.exists()
            and PAPER_EVIDENCE_TABLE_CSV_PATH.exists(),
            details={
                "json": str(PAPER_EVIDENCE_TABLE_PATH),
                "markdown": str(PAPER_EVIDENCE_TABLE_MD_PATH),
                "csv": str(PAPER_EVIDENCE_TABLE_CSV_PATH),
            },
        )
    )

    run_manifest = load_json(RUN_MANIFEST_PATH)
    run_summary = load_json(RUN_SUMMARY_PATH)
    scoring_manifest = load_json(SCORING_MANIFEST_PATH)
    utility_probe_results = load_json(UTILITY_PROBE_RESULTS_PATH)
    utility_sensitivity_audit = load_json(UTILITY_SENSITIVITY_AUDIT_PATH) if UTILITY_SENSITIVITY_AUDIT_PATH.exists() else {}
    utility_power_sweep_report = load_json(UTILITY_POWER_SWEEP_REPORT_PATH) if UTILITY_POWER_SWEEP_REPORT_PATH.exists() else {}
    curation_readiness_report = load_json(CURATION_READINESS_REPORT_PATH) if CURATION_READINESS_REPORT_PATH.exists() else {}
    stage_c_protocol_decision_report = load_json(STAGE_C_PROTOCOL_DECISION_REPORT_PATH) if STAGE_C_PROTOCOL_DECISION_REPORT_PATH.exists() else {}
    strict_baseline_control_report = load_json(STRICT_BASELINE_CONTROL_REPORT_PATH) if STRICT_BASELINE_CONTROL_REPORT_PATH.exists() else {}
    curation_decision_report = load_json(CURATION_DECISION_REPORT_PATH) if CURATION_DECISION_REPORT_PATH.exists() else {}
    paper_evidence_table = load_json(PAPER_EVIDENCE_TABLE_PATH) if PAPER_EVIDENCE_TABLE_PATH.exists() else {}
    selector_baseline_audit = load_json(SELECTOR_BASELINE_AUDIT_PATH) if SELECTOR_BASELINE_AUDIT_PATH.exists() else {}
    utility_transfer_gap_report = load_json(UTILITY_TRANSFER_GAP_REPORT_PATH) if UTILITY_TRANSFER_GAP_REPORT_PATH.exists() else {}
    core_proxy_alignment_report = load_json(CORE_PROXY_ALIGNMENT_REPORT_PATH) if CORE_PROXY_ALIGNMENT_REPORT_PATH.exists() else {}
    core_proxy_calibration_report = load_json(CORE_PROXY_CALIBRATION_REPORT_PATH) if CORE_PROXY_CALIBRATION_REPORT_PATH.exists() else {}
    anti_memorization_probe_report = load_json(ANTI_MEMORIZATION_PROBE_REPORT_PATH) if ANTI_MEMORIZATION_PROBE_REPORT_PATH.exists() else {}
    anti_memorization_probe_reports = []
    for path in sorted((Path(__file__).resolve().parent / "outputs" / "validation").glob(ANTI_MEMORIZATION_PROBE_REPORT_GLOB)):
        try:
            report = load_json(path)
        except Exception:
            continue
        if isinstance(report, dict):
            anti_memorization_probe_reports.append((path, report))
    utility_baseline_comparison_reports = []
    for path in sorted((Path(__file__).resolve().parent / "outputs" / "validation").glob(UTILITY_BASELINE_COMPARISON_GLOB)):
        try:
            report = load_json(path)
        except Exception:
            continue
        if isinstance(report, dict):
            utility_baseline_comparison_reports.append((path, report))
    utility_matching_decomposition_reports = []
    for path in sorted((Path(__file__).resolve().parent / "outputs" / "validation").glob(UTILITY_MATCHING_DECOMPOSITION_GLOB)):
        try:
            report = load_json(path)
        except Exception:
            continue
        if isinstance(report, dict):
            utility_matching_decomposition_reports.append((path, report))
    candidate_profile_comparison_reports = []
    for path in sorted((Path(__file__).resolve().parent / "outputs" / "validation").glob(CANDIDATE_PROFILE_COMPARISON_GLOB)):
        try:
            report = load_json(path)
        except Exception:
            continue
        if isinstance(report, dict):
            candidate_profile_comparison_reports.append((path, report))
    metric_spec_fingerprint = fingerprint_files([METRIC_SPEC_PATH])
    scoring_contract_fingerprint = scoring_metric_spec_fingerprint(METRIC_SPEC_PATH)

    items.append(ValidationItem(name="run_manifest_schema", ok=run_manifest.get("schema_version") == SCHEMA_VERSION, details={"schema_version": run_manifest.get("schema_version")}))
    items.append(ValidationItem(name="run_summary_schema", ok=run_summary.get("schema_version") == SCHEMA_VERSION, details={"schema_version": run_summary.get("schema_version")}))
    items.append(
        ValidationItem(
            name="run_manifest_metric_spec_fingerprint",
            ok=run_manifest.get("metric_spec_fingerprint") == metric_spec_fingerprint,
            details={"manifest": run_manifest.get("metric_spec_fingerprint"), "current": metric_spec_fingerprint},
        )
    )
    items.append(
        ValidationItem(
            name="run_summary_metric_spec_fingerprint",
            ok=run_summary.get("metric_spec_fingerprint") == metric_spec_fingerprint,
            details={"summary": run_summary.get("metric_spec_fingerprint"), "current": metric_spec_fingerprint},
        )
    )
    items.append(
        ValidationItem(
            name="scoring_manifest_scoring_metric_spec_fingerprint",
            ok=(
                scoring_manifest.get("scoring_metric_spec_fingerprint") == scoring_contract_fingerprint
                or (
                    scoring_manifest.get("scoring_metric_spec_fingerprint") is None
                    and scoring_manifest.get("metric_spec_fingerprint") == metric_spec_fingerprint
                )
            ),
            details={
                "manifest_scoring": scoring_manifest.get("scoring_metric_spec_fingerprint"),
                "current_scoring": scoring_contract_fingerprint,
                "legacy_manifest_full": scoring_manifest.get("metric_spec_fingerprint"),
                "current_full": metric_spec_fingerprint,
            },
        )
    )
    if utility_sensitivity_audit:
        audit_datasets = utility_sensitivity_audit.get("datasets") or {}
        items.append(
            ValidationItem(
                name="utility_sensitivity_audit_schema",
                ok=utility_sensitivity_audit.get("schema_version") == "utility-sensitivity-audit-v1"
                and isinstance(audit_datasets, dict),
                details={
                    "path": str(UTILITY_SENSITIVITY_AUDIT_PATH),
                    "schema_version": utility_sensitivity_audit.get("schema_version"),
                    "dataset_count": len(audit_datasets) if isinstance(audit_datasets, dict) else None,
                },
            )
        )
        expected_profile_datasets = sorted(
            str(name)
            for profile in (run_manifest.get("profiles") or {}).values()
            for name in ((profile.get("datasets") or {}).keys())
        )
        expected_profile_datasets = sorted(set(expected_profile_datasets))
        for dataset_name in expected_profile_datasets:
            payload = audit_datasets.get(dataset_name) if isinstance(audit_datasets, dict) else None
            sensitivity = (payload or {}).get("probe_sensitivity") if isinstance(payload, dict) else None
            root = (payload or {}).get("root_cause_decision") if isinstance(payload, dict) else None
            items.append(
                ValidationItem(
                    name=f"utility_sensitivity_audit_dataset_{dataset_name}",
                    ok=isinstance(payload, dict)
                    and isinstance(sensitivity, dict)
                    and isinstance(sensitivity.get("order_pass"), bool)
                    and isinstance(sensitivity.get("probe_valid"), bool)
                    and isinstance(sensitivity.get("selected_gt_random"), bool)
                    and isinstance(root, dict)
                    and isinstance(root.get("primary_hypothesis"), str)
                    and root.get("selector_tuning_allowed") is False
                    and root.get("selector_policy_action") == "hold",
                    details={"dataset": dataset_name, "probe_sensitivity": sensitivity, "root_cause_decision": root},
                )
            )
    if curation_readiness_report:
        report_datasets = curation_readiness_report.get("datasets") or {}
        allowed_action_categories = {
            "coverage_not_ready",
            "probe_power_or_control_design",
            "selector_underperforms_stageA_random",
            "feature_space_utility_transfer_gap",
            "utility_transfer_near_noise_floor",
            "utility_power_sweep_selected_not_supported",
            "lm_train_signal_gap",
            "lm_train_memorization_proxy_gap",
            "anti_memorization_probe_supports_selector",
            "utility_signal_with_token_exposure_caveat",
            "strict_counterfactual_not_ready",
            "ready_or_near_ready",
            "stage_c_ready",
            "stage_c_ready_with_token_exposure_caveat",
            "probe_preset_standardization",
        }
        allowed_framework_implication_statuses = {
            "utility_probe_not_interpretable",
            "core_policy_proxy_not_utility_supported",
            "strict_baseline_confounded_by_easy_nll_signal",
            "possible_easy_nll_baseline_confound",
            "feature_utility_transfer_gap",
            "candidate_ready_for_strict_protocol_revision",
            "strict_counterfactual_not_resolved",
            "stage_c_development_ready",
            "stage_c_development_ready_with_token_exposure_caveat",
            "utility_probe_preset_instability",
        }
        expected_summary_datasets = sorted(
            set(
                str(name)
                for profile in (run_summary.get("profiles") or {}).values()
                for name in (profile.keys() if isinstance(profile, dict) else [])
                if not str(name).startswith("_")
            )
        )
        items.append(
            ValidationItem(
                name="curation_readiness_report_schema",
                ok=curation_readiness_report.get("schema_version") == "curation-readiness-report-v1"
                and isinstance(report_datasets, dict),
                details={
                    "path": str(CURATION_READINESS_REPORT_PATH),
                    "schema_version": curation_readiness_report.get("schema_version"),
                    "dataset_count": len(report_datasets) if isinstance(report_datasets, dict) else None,
                },
            )
        )
        items.append(
            ValidationItem(
                name="curation_readiness_report_dataset_keys",
                ok=sorted(report_datasets.keys()) == expected_summary_datasets,
                details={"report": sorted(report_datasets.keys()), "expected": expected_summary_datasets},
            )
        )
        for dataset_name, payload in report_datasets.items():
            action = (payload or {}).get("recommended_next_action") if isinstance(payload, dict) else None
            utility = (payload or {}).get("utility") if isinstance(payload, dict) else None
            stage_c = (payload or {}).get("stage_c") if isinstance(payload, dict) else None
            implication = (payload or {}).get("framework_implication") if isinstance(payload, dict) else None
            items.append(
                ValidationItem(
                    name=f"curation_readiness_report_action_{dataset_name}",
                    ok=isinstance(action, dict)
                    and action.get("category") in allowed_action_categories
                    and isinstance(action.get("action"), str)
                    and isinstance(action.get("command_hint"), str)
                    and isinstance(implication, dict)
                    and implication.get("status") in allowed_framework_implication_statuses
                    and isinstance(utility, dict)
                    and isinstance(stage_c, dict),
                    details={
                        "recommended_next_action": action,
                        "framework_implication": implication,
                        "utility": utility,
                        "stage_c": stage_c,
                    },
                )
            )
    if stage_c_protocol_decision_report:
        decision_datasets = stage_c_protocol_decision_report.get("datasets") or {}
        allowed_protocol_statuses = {
            "probe_protocol_candidate_not_certified",
            "operational_total_effect_certification_candidate",
            "operational_total_effect_development_ready",
            "conditional_matched_support_without_total_effect",
            "no_operational_utility_gain",
        }
        expected_readiness_datasets = sorted((curation_readiness_report.get("datasets") or {}).keys())
        contract = stage_c_protocol_decision_report.get("framework_contract") or {}
        global_decision = stage_c_protocol_decision_report.get("global_decision") or {}
        global_power = stage_c_protocol_decision_report.get("global_power_sweep_decision") or {}
        items.append(
            ValidationItem(
                name="stage_c_protocol_decision_report_schema",
                ok=stage_c_protocol_decision_report.get("schema_version") == "stage-c-protocol-decision-report-v1"
                and isinstance(decision_datasets, dict)
                and isinstance(global_decision.get("profile_promoted"), bool)
                and isinstance(global_decision.get("global_default_utility_preset_available"), bool)
                and isinstance(global_decision.get("global_replicated_default_utility_family_available"), bool)
                and isinstance(global_decision.get("common_valid_selected_gt_random_presets"), list)
                and isinstance(global_decision.get("common_replicated_valid_families"), list)
                and isinstance(global_power.get("valid_selected_gt_random_presets_by_dataset"), dict)
                and isinstance(global_power.get("replicated_valid_families_by_dataset"), dict)
                and isinstance(global_power.get("replicated_valid_family_replicates_by_dataset"), dict)
                and isinstance(global_power.get("best_valid_selected_gt_random_preset_by_dataset"), dict)
                and isinstance(global_power.get("best_replicated_valid_preset_by_dataset"), dict)
                and isinstance(global_power.get("global_default_preset_available"), bool)
                and isinstance(global_power.get("global_replicated_default_family_available"), bool)
                and contract.get("stage_a") == "chunk-level hard gate"
                and contract.get("stage_b") == "chunk-level selection"
                and contract.get("stage_c") == "subset-level validation"
                and str(contract.get("utility_scope") or "").startswith("Stage C only")
                and contract.get("primary_utility_estimand") == "selected_vs_equal_budget_disjoint_stageA_random"
                and str(contract.get("matched_controls_role") or "").startswith("conditional mechanism diagnostics"),
                details={
                    "path": str(STAGE_C_PROTOCOL_DECISION_REPORT_PATH),
                    "schema_version": stage_c_protocol_decision_report.get("schema_version"),
                    "framework_contract": contract,
                    "global_decision": global_decision,
                    "global_power_sweep_decision": global_power,
                },
            )
        )
        items.append(
            ValidationItem(
                name="stage_c_protocol_decision_report_dataset_keys",
                ok=sorted(decision_datasets.keys()) == expected_readiness_datasets,
                details={"report": sorted(decision_datasets.keys()), "expected": expected_readiness_datasets},
            )
        )
        for dataset_name, payload in decision_datasets.items():
            items.append(
                ValidationItem(
                    name=f"stage_c_protocol_decision_report_dataset_{dataset_name}",
                    ok=isinstance(payload, dict)
                    and payload.get("protocol_status") in allowed_protocol_statuses
                    and isinstance(payload.get("decision"), str)
                    and isinstance(payload.get("next_protocol_step"), str)
                    and isinstance(payload.get("stage_c_passed"), bool)
                    and isinstance(payload.get("operational_total_effect_pass"), bool)
                    and isinstance(payload.get("coverage_passed"), bool)
                    and isinstance(payload.get("token_exposure_caveat"), bool)
                    and payload.get("primary_utility_estimand") == "selected_vs_equal_budget_disjoint_stageA_random"
                    and payload.get("matched_controls_role") == "conditional_mechanism_diagnostics_not_primary_gate"
                    and isinstance(payload.get("utility_selected_beats_stageA_random"), bool)
                    and isinstance(payload.get("utility_selected_beats_multi_matched"), bool)
                    and (
                        payload.get("best_replicated_power_sweep_preset") is None
                        or isinstance(payload.get("best_replicated_power_sweep_preset"), str)
                    )
                    and (
                        payload.get("best_replicated_power_sweep_family") is None
                        or isinstance(payload.get("best_replicated_power_sweep_family"), str)
                    )
                    and isinstance(payload.get("valid_selected_gt_random_power_sweep_presets"), list)
                    and isinstance(payload.get("replicated_valid_power_sweep_families"), list)
                    and isinstance(payload.get("replicated_valid_power_sweep_family_replicates"), dict)
                    and isinstance(payload.get("power_sweep_stable_probe_valid"), bool)
                    and isinstance(payload.get("anti_memorization_diagnostic_available"), bool)
                    and isinstance(payload.get("anti_memorization_supports_selected"), bool)
                    and payload.get("selector_policy_action") == "hold"
                    and str(payload.get("utility_scope") or "").startswith("Stage C validation only"),
                    details={"dataset": dataset_name, "protocol_decision": payload},
            )
        )
    if strict_baseline_control_report:
        control_datasets = strict_baseline_control_report.get("datasets") or {}
        allowed_control_statuses = {
            "probe_protocol_before_strict_claim",
            "development_pass_with_token_caveat",
            "operational_effect_supported_for_certification_candidate",
            "operational_effect_development_only",
            "conditional_matched_support_without_total_effect",
            "no_operational_utility_gain",
        }
        expected_protocol_datasets = sorted((stage_c_protocol_decision_report.get("datasets") or {}).keys())
        contract = strict_baseline_control_report.get("framework_contract") or {}
        items.append(
            ValidationItem(
                name="strict_baseline_control_report_schema",
                ok=strict_baseline_control_report.get("schema_version") == "strict-baseline-control-report-v1"
                and isinstance(control_datasets, dict)
                and contract.get("stage_a") == "chunk-level hard gate"
                and contract.get("stage_b") == "chunk-level selection"
                and contract.get("stage_c") == "subset-level validation"
                and str(contract.get("utility_scope") or "").startswith("Stage C only")
                and str(contract.get("anti_memorization_scope") or "") == "reported diagnostic control only"
                and contract.get("primary_utility_estimand") == "selected_vs_equal_budget_disjoint_stageA_random"
                and str(contract.get("matched_controls_role") or "").startswith("conditional mechanism diagnostics"),
                details={
                    "path": str(STRICT_BASELINE_CONTROL_REPORT_PATH),
                    "schema_version": strict_baseline_control_report.get("schema_version"),
                    "framework_contract": contract,
                },
            )
        )
        items.append(
            ValidationItem(
                name="strict_baseline_control_report_dataset_keys",
                ok=sorted(control_datasets.keys()) == expected_protocol_datasets,
                details={"report": sorted(control_datasets.keys()), "expected": expected_protocol_datasets},
            )
        )
        for dataset_name, payload in control_datasets.items():
            controls = payload.get("reported_controls") if isinstance(payload, dict) else None
            anti_evidence = payload.get("anti_memorization_evidence") if isinstance(payload, dict) else None
            certification_claim_allowed = bool((payload or {}).get("certification_claim_allowed"))
            replicated_families = (payload or {}).get("replicated_valid_power_sweep_families")
            items.append(
                ValidationItem(
                    name=f"strict_baseline_control_report_dataset_{dataset_name}",
                    ok=isinstance(payload, dict)
                    and payload.get("status") in allowed_control_statuses
                    and isinstance(payload.get("certification_claim_allowed"), bool)
                    and isinstance(payload.get("next_step"), str)
                    and isinstance(payload.get("stage_c_passed"), bool)
                    and isinstance(payload.get("coverage_passed"), bool)
                    and payload.get("primary_operational_baseline") == "baseline_stageA_random"
                    and isinstance(payload.get("primary_operational_selected_beats_stageA_random"), bool)
                    and isinstance(payload.get("operational_total_effect_pass"), bool)
                    and payload.get("matched_controls_role") == "conditional_mechanism_diagnostics_not_primary_gate"
                    and isinstance(payload.get("token_exposure_caveat"), bool)
                    and isinstance(payload.get("canonical_selected_beats_multi_matched"), bool)
                    and isinstance(payload.get("anti_memorization_diagnostic_available"), bool)
                    and isinstance(payload.get("anti_memorization_supports_selected"), bool)
                    and isinstance(anti_evidence, dict)
                    and isinstance(anti_evidence.get("available"), bool)
                    and isinstance(anti_evidence.get("supports_selected"), bool)
                    and str(anti_evidence.get("scope") or "").startswith("Stage C diagnostic only")
                    and (
                        not anti_evidence.get("available")
                        or isinstance(anti_evidence.get("delta_nll"), (int, float))
                    )
                    and isinstance(replicated_families, list)
                    and isinstance(controls, list)
                    and bool(controls)
                    and all(
                        isinstance(control, dict)
                        and isinstance(control.get("name"), str)
                        and isinstance(control.get("role"), str)
                        and isinstance(control.get("selected_beats_control"), bool)
                        and str(control.get("certification_role") or "") != "selector_objective"
                        and (
                            control.get("name") != "baseline_anti_memorization_matched_stageA_random"
                            or isinstance(control.get("delta_nll"), (int, float))
                        )
                        for control in controls
                    )
                    and any(
                        isinstance(control, dict)
                        and control.get("name") == "baseline_stageA_random"
                        and control.get("certification_role") == "primary_utility_estimand"
                        for control in controls
                    )
                    and payload.get("selector_policy_action") == "hold"
                    and str(payload.get("utility_scope") or "").startswith("Stage C validation only")
                    and (not certification_claim_allowed or bool(replicated_families)),
                    details={"dataset": dataset_name, "strict_baseline_control": payload, "anti_memorization_evidence": anti_evidence},
                )
            )
    if utility_transfer_gap_report:
        gap_datasets = utility_transfer_gap_report.get("datasets") or {}
        allowed_gap_categories = {
            "selector_feature_space_not_stronger",
            "probe_not_ready_for_transfer_claim",
            "lm_train_signal_gap",
            "lm_train_memorization_proxy_gap",
            "anti_memorization_probe_supports_selector",
            "utility_power_sweep_selected_not_supported",
            "utility_transfer_near_noise_floor",
            "eval_transfer_negative",
            "strict_counterfactual_gap",
            "stage_c_development_ready",
            "stage_c_development_ready_with_token_exposure_caveat",
            "probe_preset_candidate_available",
        }
        allowed_framework_implication_statuses = {
            "utility_probe_not_interpretable",
            "core_policy_proxy_not_utility_supported",
            "strict_baseline_confounded_by_easy_nll_signal",
            "possible_easy_nll_baseline_confound",
            "feature_utility_transfer_gap",
            "candidate_ready_for_strict_protocol_revision",
            "strict_counterfactual_not_resolved",
            "stage_c_development_ready",
            "stage_c_development_ready_with_token_exposure_caveat",
            "utility_probe_preset_instability",
        }
        expected_summary_datasets = sorted(
            set(
                str(name)
                for profile in (run_summary.get("profiles") or {}).values()
                for name in (profile.keys() if isinstance(profile, dict) else [])
                if not str(name).startswith("_")
            )
        )
        items.append(
            ValidationItem(
                name="utility_transfer_gap_report_schema",
                ok=utility_transfer_gap_report.get("schema_version") == "utility-transfer-gap-report-v1"
                and isinstance(gap_datasets, dict),
                details={
                    "path": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
                    "schema_version": utility_transfer_gap_report.get("schema_version"),
                    "dataset_count": len(gap_datasets) if isinstance(gap_datasets, dict) else None,
                },
            )
        )
        items.append(
            ValidationItem(
                name="utility_transfer_gap_report_dataset_keys",
                ok=sorted(gap_datasets.keys()) == expected_summary_datasets,
                details={"report": sorted(gap_datasets.keys()), "expected": expected_summary_datasets},
            )
        )
        for dataset_name, payload in gap_datasets.items():
            transfer_gap = (payload or {}).get("transfer_gap") if isinstance(payload, dict) else None
            feature_space = (payload or {}).get("feature_space") if isinstance(payload, dict) else None
            utility = (payload or {}).get("utility") if isinstance(payload, dict) else None
            items.append(
                ValidationItem(
                    name=f"utility_transfer_gap_report_category_{dataset_name}",
                    ok=isinstance(transfer_gap, dict)
                    and transfer_gap.get("category") in allowed_gap_categories
                    and isinstance(transfer_gap.get("action"), str)
                    and isinstance(transfer_gap.get("framework_implication"), dict)
                    and (transfer_gap.get("framework_implication") or {}).get("status")
                    in allowed_framework_implication_statuses
                    and isinstance((transfer_gap.get("framework_implication") or {}).get("selector_policy_action"), str)
                    and isinstance((transfer_gap.get("framework_implication") or {}).get("strict_baseline_action"), str)
                    and isinstance(transfer_gap.get("curation_margin"), dict)
                    and isinstance(transfer_gap.get("strict_margin"), dict)
                    and isinstance(feature_space, dict)
                    and isinstance(utility, dict),
                    details={"transfer_gap": transfer_gap, "feature_space": feature_space, "utility": utility},
                )
            )
    if curation_readiness_report and utility_transfer_gap_report:
        readiness_datasets = curation_readiness_report.get("datasets") or {}
        transfer_datasets = utility_transfer_gap_report.get("datasets") or {}
        action_categories_by_transfer = {
            "probe_preset_candidate_available": {"probe_preset_standardization"},
            "anti_memorization_probe_supports_selector": {"anti_memorization_probe_supports_selector"},
            "stage_c_development_ready": {"stage_c_ready", "stage_c_ready_with_token_exposure_caveat"},
            "stage_c_development_ready_with_token_exposure_caveat": {"stage_c_ready_with_token_exposure_caveat"},
            "utility_power_sweep_selected_not_supported": {"utility_power_sweep_selected_not_supported"},
            "strict_counterfactual_gap": {"strict_counterfactual_not_ready"},
        }
        items.append(
            ValidationItem(
                name="curation_readiness_transfer_gap_consistent_dataset_keys",
                ok=sorted(readiness_datasets.keys()) == sorted(transfer_datasets.keys()),
                details={"curation": sorted(readiness_datasets.keys()), "transfer": sorted(transfer_datasets.keys())},
            )
        )
        for dataset_name in sorted(set(readiness_datasets) & set(transfer_datasets)):
            readiness_payload = readiness_datasets.get(dataset_name) or {}
            transfer_payload = transfer_datasets.get(dataset_name) or {}
            readiness_gap = readiness_payload.get("utility_transfer_gap") or {}
            transfer_gap = transfer_payload.get("transfer_gap") or {}
            readiness_action = readiness_payload.get("recommended_next_action") or {}
            transfer_category = transfer_gap.get("category")
            expected_action_categories = action_categories_by_transfer.get(str(transfer_category), set())
            items.append(
                ValidationItem(
                    name=f"curation_readiness_transfer_gap_consistent_{dataset_name}",
                    ok=isinstance(readiness_gap, dict)
                    and isinstance(transfer_gap, dict)
                    and readiness_gap.get("category") == transfer_gap.get("category")
                    and _framework_status(readiness_payload) == _framework_status({"framework_implication": transfer_gap.get("framework_implication")})
                    and readiness_action.get("transfer_gap_category") in {None, transfer_gap.get("category")}
                    and (
                        not expected_action_categories
                        or readiness_action.get("category") in expected_action_categories
                    ),
                    details={
                        "dataset": dataset_name,
                        "curation_transfer_category": readiness_gap.get("category") if isinstance(readiness_gap, dict) else None,
                        "transfer_gap_category": transfer_gap.get("category") if isinstance(transfer_gap, dict) else None,
                        "curation_framework_status": _framework_status(readiness_payload),
                        "transfer_framework_status": _framework_status({"framework_implication": transfer_gap.get("framework_implication")}),
                        "recommended_next_action": readiness_action,
                        "expected_action_categories": sorted(expected_action_categories),
                    },
                )
            )
    if stage_c_protocol_decision_report and curation_readiness_report:
        protocol_datasets = stage_c_protocol_decision_report.get("datasets") or {}
        readiness_datasets = curation_readiness_report.get("datasets") or {}
        items.append(
            ValidationItem(
                name="stage_c_protocol_curation_consistent_dataset_keys",
                ok=sorted(protocol_datasets.keys()) == sorted(readiness_datasets.keys()),
                details={"protocol": sorted(protocol_datasets.keys()), "curation": sorted(readiness_datasets.keys())},
            )
        )
        for dataset_name in sorted(set(protocol_datasets) & set(readiness_datasets)):
            protocol_payload = protocol_datasets.get(dataset_name) or {}
            readiness_payload = readiness_datasets.get(dataset_name) or {}
            readiness_stage_c = readiness_payload.get("stage_c") or {}
            readiness_utility = readiness_payload.get("utility") or {}
            readiness_action = readiness_payload.get("recommended_next_action") or {}
            items.append(
                ValidationItem(
                    name=f"stage_c_protocol_curation_consistent_{dataset_name}",
                    ok=_framework_status(protocol_payload) == _framework_status(readiness_payload)
                    and protocol_payload.get("recommended_action_category") == readiness_action.get("category")
                    and protocol_payload.get("stage_c_passed") == readiness_stage_c.get("passed")
                    and protocol_payload.get("coverage_passed") == readiness_stage_c.get("coverage_pass")
                    and protocol_payload.get("token_exposure_caveat")
                    == bool(
                        readiness_utility.get("token_exposure_confounded")
                        or readiness_utility.get("token_exposure_inconclusive")
                    )
                    and protocol_payload.get("utility_selected_beats_stageA_random")
                    == readiness_utility.get("selected_beats_stageA_random")
                    and protocol_payload.get("utility_selected_beats_multi_matched")
                    == readiness_utility.get("selected_beats_multi_matched"),
                    details={
                        "dataset": dataset_name,
                        "protocol": protocol_payload,
                        "curation_framework_status": _framework_status(readiness_payload),
                        "curation_stage_c": readiness_stage_c,
                        "curation_utility": readiness_utility,
                        "recommended_next_action": readiness_action,
                    },
                )
            )
    if stage_c_protocol_decision_report and utility_transfer_gap_report:
        protocol_datasets = stage_c_protocol_decision_report.get("datasets") or {}
        transfer_datasets = utility_transfer_gap_report.get("datasets") or {}
        items.append(
            ValidationItem(
                name="stage_c_protocol_transfer_gap_consistent_dataset_keys",
                ok=sorted(protocol_datasets.keys()) == sorted(transfer_datasets.keys()),
                details={"protocol": sorted(protocol_datasets.keys()), "transfer": sorted(transfer_datasets.keys())},
            )
        )
        for dataset_name in sorted(set(protocol_datasets) & set(transfer_datasets)):
            protocol_payload = protocol_datasets.get(dataset_name) or {}
            transfer_payload = transfer_datasets.get(dataset_name) or {}
            transfer_gap = transfer_payload.get("transfer_gap") or {}
            transfer_anti = transfer_gap.get("anti_memorization_diagnostic_baseline") if isinstance(transfer_gap, dict) else {}
            if not isinstance(transfer_anti, dict):
                transfer_anti = {}
            transfer_anti_available = bool(transfer_anti.get("available"))
            items.append(
                ValidationItem(
                    name=f"stage_c_protocol_transfer_gap_consistent_{dataset_name}",
                    ok=protocol_payload.get("anti_memorization_diagnostic_available") == transfer_anti_available
                    and protocol_payload.get("anti_memorization_supports_selected")
                    == bool(transfer_anti.get("supports_selected"))
                    and (
                        not transfer_anti_available
                        or protocol_payload.get("anti_memorization_delta_nll") == transfer_anti.get("delta_nll")
                    ),
                    details={
                        "dataset": dataset_name,
                        "protocol_anti_available": protocol_payload.get("anti_memorization_diagnostic_available"),
                        "protocol_anti_supports": protocol_payload.get("anti_memorization_supports_selected"),
                        "protocol_anti_delta_nll": protocol_payload.get("anti_memorization_delta_nll"),
                        "transfer_anti_memorization": transfer_anti,
                    },
                )
            )
    if stage_c_protocol_decision_report and utility_power_sweep_report:
        protocol_datasets = stage_c_protocol_decision_report.get("datasets") or {}
        global_power = stage_c_protocol_decision_report.get("global_power_sweep_decision") or {}
        global_decision = stage_c_protocol_decision_report.get("global_decision") or {}
        protocol_profile = str(stage_c_protocol_decision_report.get("profile") or "")
        power_sweep_profile = str(utility_power_sweep_report.get("profile") or "")
        power_sweep_profile_matches = bool(protocol_profile and power_sweep_profile == protocol_profile)
        scoped_power_sweep = utility_power_sweep_report if power_sweep_profile_matches else {}
        items.append(
            ValidationItem(
                name="stage_c_protocol_power_sweep_profile_scope",
                ok=stage_c_protocol_decision_report.get("power_sweep_profile") == utility_power_sweep_report.get("profile")
                and stage_c_protocol_decision_report.get("power_sweep_profile_matches") == power_sweep_profile_matches,
                details={
                    "protocol_profile": protocol_profile,
                    "power_sweep_profile": power_sweep_profile,
                    "reported_profile_matches": stage_c_protocol_decision_report.get("power_sweep_profile_matches"),
                },
            )
        )
        computed_valid_by_dataset = {
            dataset_name: _as_str_list(_power_sweep_decision(scoped_power_sweep, dataset_name).get("valid_selected_gt_random_presets"))
            for dataset_name in sorted(protocol_datasets.keys())
        }
        computed_replicated_by_dataset = {
            dataset_name: _replicated_valid_families(scoped_power_sweep, dataset_name)
            for dataset_name in computed_valid_by_dataset
        }
        computed_replicated_replicates_by_dataset = {
            dataset_name: _replicated_valid_family_replicates(scoped_power_sweep, dataset_name)
            for dataset_name in computed_valid_by_dataset
        }
        computed_best_by_dataset = {
            dataset_name: _power_sweep_decision(scoped_power_sweep, dataset_name).get("best_valid_selected_gt_random_preset")
            for dataset_name in sorted(protocol_datasets.keys())
        }
        computed_best_replicated_by_dataset = {
            dataset_name: _best_replicated_preset(scoped_power_sweep, dataset_name)
            for dataset_name in sorted(protocol_datasets.keys())
        }
        computed_best_replicated_family_by_dataset = {
            dataset_name: _best_replicated_family(scoped_power_sweep, dataset_name)
            for dataset_name in sorted(protocol_datasets.keys())
        }
        valid_sets = [set(presets) for presets in computed_valid_by_dataset.values() if presets]
        computed_common_valid = (
            sorted(set.intersection(*valid_sets))
            if len(valid_sets) == len(computed_valid_by_dataset) and valid_sets
            else []
        )
        replicated_sets = [set(families) for families in computed_replicated_by_dataset.values() if families]
        computed_common_replicated = (
            sorted(set.intersection(*replicated_sets))
            if len(replicated_sets) == len(computed_replicated_by_dataset) and replicated_sets
            else []
        )
        for dataset_name, protocol_payload in protocol_datasets.items():
            power_decision = _power_sweep_decision(scoped_power_sweep, dataset_name)
            valid_presets = computed_valid_by_dataset.get(dataset_name) or []
            items.append(
                ValidationItem(
                    name=f"stage_c_protocol_power_sweep_consistent_{dataset_name}",
                    ok=_as_str_list(protocol_payload.get("valid_selected_gt_random_power_sweep_presets")) == valid_presets
                    and protocol_payload.get("best_valid_power_sweep_preset") == computed_best_by_dataset.get(dataset_name)
                    and protocol_payload.get("best_replicated_power_sweep_preset")
                    == computed_best_replicated_by_dataset.get(dataset_name)
                    and protocol_payload.get("best_replicated_power_sweep_family")
                    == computed_best_replicated_family_by_dataset.get(dataset_name)
                    and _as_str_list(protocol_payload.get("replicated_valid_power_sweep_families"))
                    == computed_replicated_by_dataset.get(dataset_name)
                    and protocol_payload.get("replicated_valid_power_sweep_family_replicates")
                    == computed_replicated_replicates_by_dataset.get(dataset_name)
                    and protocol_payload.get("power_sweep_stable_probe_valid") == bool(power_decision.get("stable_probe_valid")),
                    details={
                        "dataset": dataset_name,
                        "protocol_valid_presets": protocol_payload.get("valid_selected_gt_random_power_sweep_presets"),
                        "power_valid_presets": valid_presets,
                        "protocol_best_preset": protocol_payload.get("best_valid_power_sweep_preset"),
                        "power_best_preset": computed_best_by_dataset.get(dataset_name),
                        "protocol_best_replicated_preset": protocol_payload.get("best_replicated_power_sweep_preset"),
                        "computed_best_replicated_preset": computed_best_replicated_by_dataset.get(dataset_name),
                        "protocol_best_replicated_family": protocol_payload.get("best_replicated_power_sweep_family"),
                        "computed_best_replicated_family": computed_best_replicated_family_by_dataset.get(dataset_name),
                        "protocol_replicated_families": protocol_payload.get("replicated_valid_power_sweep_families"),
                        "computed_replicated_families": computed_replicated_by_dataset.get(dataset_name),
                        "protocol_replicated_family_replicates": protocol_payload.get("replicated_valid_power_sweep_family_replicates"),
                        "computed_replicated_family_replicates": computed_replicated_replicates_by_dataset.get(dataset_name),
                        "power_decision": power_decision,
                    },
                )
            )
        items.append(
            ValidationItem(
                name="stage_c_protocol_global_power_sweep_consistent",
                ok=global_power.get("valid_selected_gt_random_presets_by_dataset") == computed_valid_by_dataset
                and global_power.get("replicated_valid_families_by_dataset") == computed_replicated_by_dataset
                and global_power.get("replicated_valid_family_replicates_by_dataset")
                == computed_replicated_replicates_by_dataset
                and global_power.get("best_valid_selected_gt_random_preset_by_dataset") == computed_best_by_dataset
                and global_power.get("best_replicated_valid_preset_by_dataset")
                == computed_best_replicated_by_dataset
                and global_power.get("common_valid_selected_gt_random_presets") == computed_common_valid
                and global_power.get("common_replicated_valid_families") == computed_common_replicated
                and global_power.get("global_default_preset_available") == bool(computed_common_valid)
                and global_power.get("global_replicated_default_family_available") == bool(computed_common_replicated)
                and global_decision.get("common_valid_selected_gt_random_presets") == computed_common_valid
                and global_decision.get("common_replicated_valid_families") == computed_common_replicated
                and global_decision.get("global_default_utility_preset_available") == bool(computed_common_valid)
                and global_decision.get("global_replicated_default_utility_family_available")
                == bool(computed_common_replicated),
                details={
                    "computed_valid_by_dataset": computed_valid_by_dataset,
                    "computed_replicated_by_dataset": computed_replicated_by_dataset,
                    "computed_replicated_replicates_by_dataset": computed_replicated_replicates_by_dataset,
                    "computed_best_by_dataset": computed_best_by_dataset,
                    "computed_best_replicated_by_dataset": computed_best_replicated_by_dataset,
                    "computed_common_valid": computed_common_valid,
                    "computed_common_replicated": computed_common_replicated,
                    "global_power_sweep_decision": global_power,
                    "global_decision": global_decision,
                },
            )
        )
    if strict_baseline_control_report and stage_c_protocol_decision_report:
        strict_datasets = strict_baseline_control_report.get("datasets") or {}
        protocol_datasets = stage_c_protocol_decision_report.get("datasets") or {}
        readiness_datasets = (curation_readiness_report.get("datasets") or {}) if curation_readiness_report else {}
        transfer_datasets = (utility_transfer_gap_report.get("datasets") or {}) if utility_transfer_gap_report else {}
        items.append(
            ValidationItem(
                name="strict_baseline_protocol_consistent_dataset_keys",
                ok=sorted(strict_datasets.keys()) == sorted(protocol_datasets.keys()),
                details={"strict": sorted(strict_datasets.keys()), "protocol": sorted(protocol_datasets.keys())},
            )
        )
        for dataset_name in sorted(set(strict_datasets) & set(protocol_datasets)):
            strict_payload = strict_datasets.get(dataset_name) or {}
            protocol_payload = protocol_datasets.get(dataset_name) or {}
            readiness_payload = readiness_datasets.get(dataset_name) or {}
            transfer_payload = transfer_datasets.get(dataset_name) or {}
            transfer_gap = transfer_payload.get("transfer_gap") or {}
            anti_evidence = strict_payload.get("anti_memorization_evidence") or {}
            transfer_anti = transfer_gap.get("anti_memorization_diagnostic_baseline") if isinstance(transfer_gap, dict) else {}
            if not isinstance(transfer_anti, dict):
                transfer_anti = {}
            items.append(
                ValidationItem(
                    name=f"strict_baseline_protocol_consistent_{dataset_name}",
                    ok=_framework_status(strict_payload) == _framework_status(protocol_payload)
                    and (not readiness_payload or _framework_status(strict_payload) == _framework_status(readiness_payload))
                    and strict_payload.get("protocol_status") == protocol_payload.get("protocol_status")
                    and strict_payload.get("stage_c_passed") == protocol_payload.get("stage_c_passed")
                    and strict_payload.get("token_exposure_caveat") == protocol_payload.get("token_exposure_caveat")
                    and strict_payload.get("replicated_valid_power_sweep_families")
                    == protocol_payload.get("replicated_valid_power_sweep_families")
                    and strict_payload.get("anti_memorization_diagnostic_available")
                    == protocol_payload.get("anti_memorization_diagnostic_available")
                    and strict_payload.get("anti_memorization_supports_selected")
                    == protocol_payload.get("anti_memorization_supports_selected")
                    and strict_payload.get("selector_policy_action") == protocol_payload.get("selector_policy_action")
                    and (
                        not transfer_gap
                        or strict_payload.get("transfer_gap_category") == transfer_gap.get("category")
                    )
                    and (
                        not anti_evidence.get("available")
                        or anti_evidence.get("supports_selected") == protocol_payload.get("anti_memorization_supports_selected")
                    )
                    and (
                        not transfer_anti
                        or anti_evidence.get("delta_nll") == transfer_anti.get("delta_nll")
                    ),
                    details={
                        "dataset": dataset_name,
                        "strict": strict_payload,
                        "protocol": protocol_payload,
                        "curation_framework_status": _framework_status(readiness_payload) if readiness_payload else None,
                        "transfer_gap_category": transfer_gap.get("category") if isinstance(transfer_gap, dict) else None,
                        "transfer_anti_memorization": transfer_anti,
                    },
                )
            )
    if curation_decision_report and strict_baseline_control_report:
        decision_datasets = curation_decision_report.get("datasets") or {}
        strict_datasets = strict_baseline_control_report.get("datasets") or {}
        protocol_datasets = (stage_c_protocol_decision_report.get("datasets") or {}) if stage_c_protocol_decision_report else {}
        readiness_datasets = (curation_readiness_report.get("datasets") or {}) if curation_readiness_report else {}
        allowed_decisions = {
            "accepted_for_training",
            "accepted_for_training_with_caveat",
            "needs_certification_utility",
            "utility_probe_unstable",
            "rejected_for_training",
            "insufficient_usable_data",
        }
        allowed_training_use = {
            "certification_candidate",
            "development_only_with_token_exposure_caveat",
            "hold_for_stage_c_protocol_standardization",
            "development_only",
            "do_not_use_without_followup",
            "do_not_train_insufficient_usable_data",
        }
        allowed_operational_actions = {
            "accept",
            "accept_with_caveat",
            "manual_review",
            "reject",
            "insufficient_usable_data",
        }

        def _expected_curation_decision(
            readiness_payload: Dict[str, Any],
            protocol_payload: Dict[str, Any],
            strict_payload: Dict[str, Any],
            usable_data_sufficient: bool,
        ) -> str:
            stage_c = readiness_payload.get("stage_c") or {}
            utility = readiness_payload.get("utility") or {}
            protocol_status = str(protocol_payload.get("protocol_status") or "")
            strict_status = str(strict_payload.get("status") or "")
            token_caveat = bool(strict_payload.get("token_exposure_caveat") or protocol_payload.get("token_exposure_caveat"))
            selected_beats_random = bool(
                utility.get("selected_beats_stageA_random")
                or strict_payload.get("primary_operational_selected_beats_stageA_random")
            )
            coverage_passed = bool(stage_c.get("coverage_pass") or strict_payload.get("coverage_passed"))
            operational_total_effect_pass = bool(strict_payload.get("operational_total_effect_pass") or (coverage_passed and selected_beats_random))
            if not usable_data_sufficient:
                return "insufficient_usable_data"
            if bool(strict_payload.get("certification_claim_allowed")):
                return "accepted_for_training"
            if token_caveat and operational_total_effect_pass:
                return "needs_certification_utility"
            if protocol_status == "probe_protocol_candidate_not_certified":
                return "utility_probe_unstable"
            if strict_status == "conditional_matched_support_without_total_effect":
                return "rejected_for_training"
            if operational_total_effect_pass:
                return "accepted_for_training_with_caveat"
            return "rejected_for_training"

        items.append(
            ValidationItem(
                name="curation_decision_report_schema",
                ok=curation_decision_report.get("schema_version") == "curation-decision-report-v2"
                and isinstance(decision_datasets, dict)
                and ((curation_decision_report.get("research_framing") or {}).get("output") == "curated LM-training dataset or explicit abstention")
                and ((curation_decision_report.get("framework_contract") or {}).get("decision_layer") == "training-use claim over the selected subset"),
                details={
                    "path": str(CURATION_DECISION_REPORT_PATH),
                    "schema_version": curation_decision_report.get("schema_version"),
                    "research_framing": curation_decision_report.get("research_framing"),
                    "framework_contract": curation_decision_report.get("framework_contract"),
                },
            )
        )
        items.append(
            ValidationItem(
                name="curation_decision_report_dataset_keys",
                ok=sorted(decision_datasets.keys()) == sorted(strict_datasets.keys()),
                details={"decision": sorted(decision_datasets.keys()), "strict": sorted(strict_datasets.keys())},
            )
        )
        computed_decision_counts: Dict[str, int] = {}
        computed_caveat_counts: Dict[str, int] = {}
        for dataset_name, payload in decision_datasets.items():
            if not isinstance(payload, dict):
                payload = {}
            strict_payload = strict_datasets.get(dataset_name) or {}
            protocol_payload = protocol_datasets.get(dataset_name) or {}
            readiness_payload = readiness_datasets.get(dataset_name) or {}
            evidence_matrix = payload.get("evidence_matrix") if isinstance(payload, dict) else None
            if not isinstance(evidence_matrix, list):
                evidence_matrix = []
            stage_labels = [str(item.get("stage") or "") for item in evidence_matrix if isinstance(item, dict)]
            stage_a_items = [
                item for item in evidence_matrix
                if isinstance(item, dict) and str(item.get("stage") or "") == "A"
            ]
            stage_a_values = (stage_a_items[0].get("evidence") or {}) if stage_a_items else {}
            independently_sufficient = bool(
                int(stage_a_values.get("stage_a_records") or 0) > 0
                and int(stage_a_values.get("selected_records") or 0) > 0
                and int(stage_a_values.get("stage_a_records") or 0) >= int(stage_a_values.get("selected_records") or 0)
                and int(stage_a_values.get("stage_a_candidate_records_excluding_selected") or 0) > 0
            )
            expected_decision = _expected_curation_decision(
                readiness_payload,
                protocol_payload,
                strict_payload,
                independently_sufficient,
            )
            decision = str(payload.get("decision") or "")
            computed_decision_counts[decision] = computed_decision_counts.get(decision, 0) + 1
            for caveat in payload.get("caveats") or []:
                computed_caveat_counts[str(caveat)] = computed_caveat_counts.get(str(caveat), 0) + 1
            items.append(
                ValidationItem(
                    name=f"curation_decision_report_dataset_{dataset_name}",
                    ok=decision in allowed_decisions
                    and payload.get("training_use") in allowed_training_use
                    and payload.get("operational_action") in allowed_operational_actions
                    and payload.get("usable_data_sufficient") == independently_sufficient
                    and payload.get("certification_claim_allowed")
                    == (bool(strict_payload.get("certification_claim_allowed")) and independently_sufficient)
                    and payload.get("utility_scope") == "Stage C validation only; never selector objective"
                    and decision == expected_decision
                    and isinstance(payload.get("rationale"), str)
                    and bool(payload.get("rationale"))
                    and isinstance(payload.get("next_step"), str)
                    and bool(payload.get("next_step"))
                    and {"A", "B", "C"}.issubset(set(stage_labels))
                    and len(evidence_matrix) >= 6
                    and all(
                        isinstance(item, dict)
                        and item.get("status") in {"pass", "fail", "pass_with_caveat", "missing"}
                        and isinstance(item.get("claim"), str)
                        and isinstance(item.get("interpretation"), str)
                        for item in evidence_matrix
                    ),
                    details={
                        "dataset": dataset_name,
                        "decision": payload.get("decision"),
                        "expected_decision": expected_decision,
                        "training_use": payload.get("training_use"),
                        "operational_action": payload.get("operational_action"),
                        "usable_data_sufficient": payload.get("usable_data_sufficient"),
                        "strict_status": strict_payload.get("status"),
                        "protocol_status": protocol_payload.get("protocol_status"),
                        "stage_labels": stage_labels,
                        "caveats": payload.get("caveats"),
                    },
                )
            )
        summary = curation_decision_report.get("summary") or {}
        items.append(
            ValidationItem(
                name="curation_decision_report_summary_counts",
                ok=summary.get("dataset_count") == len(decision_datasets)
                and summary.get("decision_counts") == computed_decision_counts
                and summary.get("caveat_counts") == computed_caveat_counts
                and summary.get("certification_claim_allowed_dataset_count")
                == sum(1 for payload in decision_datasets.values() if bool((payload or {}).get("certification_claim_allowed"))),
                details={
                    "summary": summary,
                    "computed_decision_counts": computed_decision_counts,
                    "computed_caveat_counts": computed_caveat_counts,
                },
            )
        )
    if paper_evidence_table and curation_decision_report:
        paper_datasets = paper_evidence_table.get("datasets") or {}
        decision_datasets = curation_decision_report.get("datasets") or {}
        expected_candidates = sorted(
            dataset
            for dataset, payload in decision_datasets.items()
            if isinstance(payload, dict) and bool(payload.get("certification_claim_allowed"))
        )
        paper_summary = paper_evidence_table.get("summary") or {}
        claim_boundary = paper_evidence_table.get("claim_boundary") or {}
        items.append(
            ValidationItem(
                name="paper_evidence_table_schema",
                ok=paper_evidence_table.get("schema_version") == "paper-evidence-table-v1"
                and paper_evidence_table.get("profile") == curation_decision_report.get("profile")
                and sorted(paper_datasets.keys()) == sorted(decision_datasets.keys())
                and claim_boundary.get("utility_scope") == "Stage C validation only; never selector objective",
                details={
                    "profile": paper_evidence_table.get("profile"),
                    "datasets": sorted(paper_datasets.keys()),
                    "claim_boundary": claim_boundary,
                },
            )
        )
        items.append(
            ValidationItem(
                name="paper_evidence_table_certification_candidates",
                ok=sorted(paper_summary.get("certification_candidates") or []) == expected_candidates
                and int(paper_summary.get("certification_candidate_count") or 0) == len(expected_candidates),
                details={
                    "paper_candidates": sorted(paper_summary.get("certification_candidates") or []),
                    "expected_candidates": expected_candidates,
                },
            )
        )
        for dataset_name, decision_payload in decision_datasets.items():
            paper_payload = paper_datasets.get(dataset_name) or {}
            utility = paper_payload.get("utility") or {}
            items.append(
                ValidationItem(
                    name=f"paper_evidence_table_dataset_{dataset_name}",
                    ok=paper_payload.get("decision") == decision_payload.get("decision")
                    and paper_payload.get("training_use") == decision_payload.get("training_use")
                    and paper_payload.get("certification_claim_allowed")
                    == decision_payload.get("certification_claim_allowed")
                    and paper_payload.get("selector_policy_action") == "hold"
                    and paper_payload.get("utility_scope") == "Stage C validation only; never selector objective"
                    and isinstance(utility.get("selected_beats_stage_a_random"), bool)
                    and isinstance(utility.get("selected_beats_multi_matched"), bool),
                    details={
                        "dataset": dataset_name,
                        "paper_decision": paper_payload.get("decision"),
                        "decision_report": decision_payload.get("decision"),
                        "selector_policy_action": paper_payload.get("selector_policy_action"),
                    },
                )
            )
    if core_proxy_alignment_report:
        alignment_datasets = core_proxy_alignment_report.get("datasets") or {}
        allowed_alignment_statuses = {
            "not_diagnosable_until_probe_valid",
            "core_proxy_utility_mismatch_with_easy_nll_tension",
            "core_proxy_utility_mismatch",
            "strict_baseline_easy_nll_confound_supported",
            "easy_nll_confound_candidate",
            "core_proxy_partially_supported_by_utility",
            "alignment_unresolved",
            "stage_c_development_ready",
            "stage_c_development_ready_with_token_exposure_caveat",
            "probe_preset_instability_with_candidate",
        }
        expected_transfer_datasets = sorted((utility_transfer_gap_report.get("datasets") or {}).keys())
        items.append(
            ValidationItem(
                name="core_proxy_alignment_report_schema",
                ok=core_proxy_alignment_report.get("schema_version") == "core-proxy-alignment-report-v1"
                and isinstance(alignment_datasets, dict),
                details={
                    "path": str(CORE_PROXY_ALIGNMENT_REPORT_PATH),
                    "schema_version": core_proxy_alignment_report.get("schema_version"),
                    "dataset_count": len(alignment_datasets) if isinstance(alignment_datasets, dict) else None,
                },
            )
        )
        items.append(
            ValidationItem(
                name="core_proxy_alignment_report_dataset_keys",
                ok=sorted(alignment_datasets.keys()) == expected_transfer_datasets,
                details={"report": sorted(alignment_datasets.keys()), "expected": expected_transfer_datasets},
            )
        )
        for dataset_name, payload in alignment_datasets.items():
            alignment = (payload or {}).get("alignment") if isinstance(payload, dict) else None
            easy_nll = (payload or {}).get("easy_nll_tension") if isinstance(payload, dict) else None
            verdicts = (payload or {}).get("selector_feature_verdicts") if isinstance(payload, dict) else None
            items.append(
                ValidationItem(
                    name=f"core_proxy_alignment_report_dataset_{dataset_name}",
                    ok=isinstance(alignment, dict)
                    and alignment.get("status") in allowed_alignment_statuses
                    and isinstance(alignment.get("selector_policy_action"), str)
                    and isinstance(alignment.get("next_step"), str)
                    and isinstance(easy_nll, dict)
                    and isinstance(easy_nll.get("candidate"), bool)
                    and str((payload or {}).get("selector_objective_scope") or "").startswith("Core metrics only")
                    and isinstance(verdicts, dict),
                    details={"alignment": alignment, "easy_nll_tension": easy_nll, "selector_feature_verdicts": verdicts},
                )
            )
    if core_proxy_calibration_report:
        calibration_datasets = core_proxy_calibration_report.get("datasets") or {}
        allowed_calibration_targets = {
            "probe_before_core_calibration",
            "strict_baseline_control_before_core_calibration",
            "learnability_proxy_semantics",
            "redundancy_useful_recurrence_calibration",
            "length_bucket_and_useful_length_support",
            "no_immediate_core_proxy_change",
            "stage_c_certification_followup",
            "probe_preset_standardization",
        }
        expected_alignment_datasets = sorted((core_proxy_alignment_report.get("datasets") or {}).keys())
        items.append(
            ValidationItem(
                name="core_proxy_calibration_report_schema",
                ok=core_proxy_calibration_report.get("schema_version") == "core-proxy-calibration-report-v1"
                and isinstance(calibration_datasets, dict),
                details={
                    "path": str(CORE_PROXY_CALIBRATION_REPORT_PATH),
                    "schema_version": core_proxy_calibration_report.get("schema_version"),
                    "dataset_count": len(calibration_datasets) if isinstance(calibration_datasets, dict) else None,
                },
            )
        )
        items.append(
            ValidationItem(
                name="core_proxy_calibration_report_dataset_keys",
                ok=sorted(calibration_datasets.keys()) == expected_alignment_datasets,
                details={"report": sorted(calibration_datasets.keys()), "expected": expected_alignment_datasets},
            )
        )
        for dataset_name, payload in calibration_datasets.items():
            targets = (payload or {}).get("calibration_targets") if isinstance(payload, dict) else None
            recommendation = (payload or {}).get("candidate_variant_recommendation") if isinstance(payload, dict) else None
            items.append(
                ValidationItem(
                    name=f"core_proxy_calibration_report_dataset_{dataset_name}",
                    ok=isinstance(targets, list)
                    and bool(targets)
                    and str((payload or {}).get("selector_objective_scope") or "").startswith("Core metrics only")
                    and isinstance(recommendation, dict)
                    and isinstance(recommendation.get("available"), bool)
                    and (
                        recommendation.get("recommended_variant") is None
                        or isinstance(recommendation.get("recommended_variant"), str)
                    )
                    and all(
                        isinstance(target, dict)
                        and target.get("target") in allowed_calibration_targets
                        and isinstance(target.get("priority"), int)
                        and isinstance(target.get("reason"), str)
                        and isinstance(target.get("next_experiment"), str)
                        for target in targets
                    ),
                    details={"calibration_targets": targets, "candidate_variant_recommendation": recommendation},
                )
            )
    for report_path, report_payload in anti_memorization_probe_reports:
        utility_result = report_payload.get("utility_result") or {}
        pool_diagnostics = report_payload.get("pool_diagnostics") or {}
        causal = utility_result.get("causal_utility_audit") or {}
        pool_summary = {
            "bucket_count": pool_diagnostics.get("bucket_count"),
            "selected_reference_count": pool_diagnostics.get("selected_reference_count"),
            "baseline_reference_count": pool_diagnostics.get("baseline_reference_count"),
            "matched_pool_count": pool_diagnostics.get("matched_pool_count"),
            "pool_multiplier": pool_diagnostics.get("pool_multiplier"),
            "exclude_selected": pool_diagnostics.get("exclude_selected"),
            "excluded_selected_records": pool_diagnostics.get("excluded_selected_records"),
        }
        items.append(
            ValidationItem(
                name=f"anti_memorization_probe_report_schema_{report_payload.get('dataset') or report_path.stem}",
                ok=report_payload.get("schema_version") == "anti-memorization-probe-report-v1"
                and report_payload.get("selector_objective_scope") == "diagnostic_only_not_selector_objective"
                and (
                    report_payload.get("probe_override_scope") in {None, "diagnostic_only_not_certification"}
                )
                and report_payload.get("baseline") == "baseline_anti_memorization_matched_stageA_random"
                and (
                    report_payload.get("probe_protocol") is None
                    or isinstance(report_payload.get("probe_protocol"), dict)
                )
                and isinstance(pool_diagnostics.get("matched_pool_count"), int)
                and int(pool_diagnostics.get("matched_pool_count") or 0) > 0
                and isinstance(utility_result.get("delta_nll"), (int, float))
                and isinstance(utility_result.get("delta_nll_ci_low"), (int, float))
                and isinstance(causal.get("dominant_failure_mode"), str),
                details={
                    "path": str(report_path),
                    "dataset": report_payload.get("dataset"),
                    "baseline": report_payload.get("baseline"),
                    "pool_diagnostics": pool_summary,
                    "delta_nll": utility_result.get("delta_nll"),
                    "delta_nll_ci_low": utility_result.get("delta_nll_ci_low"),
                    "causal_utility_audit": causal,
                },
            )
        )
    expected_utility_comparison_baselines = {
        "baseline_multi_matched_stageA_random",
        "baseline_nuisance_matched_stageA_random",
        "baseline_anti_memorization_matched_stageA_random",
    }
    for report_path, report_payload in utility_baseline_comparison_reports:
        summaries = report_payload.get("result_summaries") or {}
        fidelity = report_payload.get("pool_fidelity") or {}
        protocol = report_payload.get("probe_protocol") or {}
        baseline_names = set(summaries) if isinstance(summaries, dict) else set()
        run_counts = {
            int(payload.get("run_count") or 0)
            for payload in summaries.values()
            if isinstance(payload, dict)
        }
        common_summary_contract = all(
            isinstance(payload, dict)
            and isinstance(payload.get("delta_nll"), (int, float))
            and isinstance(payload.get("delta_nll_ci_low"), (int, float))
            and isinstance(payload.get("minimum_detectable_delta_nll_95_max"), (int, float))
            and isinstance(payload.get("positive_run_fraction"), (int, float))
            and int(payload.get("run_count") or 0) > 0
            for payload in summaries.values()
        )
        common_fidelity_contract = all(
            isinstance(payload, dict)
            and bool(payload.get("exclude_selected"))
            and int(payload.get("matched_pool_count") or 0) > 0
            and isinstance(payload.get("exact_bucket_availability_ratio"), (int, float))
            for payload in fidelity.values()
        )
        nuisance = fidelity.get("baseline_nuisance_matched_stageA_random") or {}
        anti = fidelity.get("baseline_anti_memorization_matched_stageA_random") or {}
        items.append(
            ValidationItem(
                name=f"utility_baseline_comparison_contract_{report_payload.get('dataset') or report_path.stem}",
                ok=report_payload.get("schema_version") == "utility-baseline-comparison-v1"
                and report_payload.get("selector_objective_scope") == "Stage C validation only; never selector objective"
                and report_payload.get("canonical_baseline") == "baseline_multi_matched_stageA_random"
                and baseline_names == expected_utility_comparison_baselines
                and set(fidelity) == expected_utility_comparison_baselines
                and common_summary_contract
                and common_fidelity_contract
                and len(run_counts) == 1
                and next(iter(run_counts), 0) > 1
                and isinstance(protocol.get("holdout_buckets"), list)
                and len(protocol.get("holdout_buckets") or []) > 1
                and isinstance(protocol.get("seeds"), list)
                and len(protocol.get("seeds") or []) > 1
                and nuisance.get("matched_variables") == ["length", "style", "domain", "repeat_pressure"]
                and nuisance.get("excluded_selector_target_variables") == ["quality", "redundancy_risk"]
                and nuisance.get("fallback_order") == []
                and anti.get("matched_variables") == ["quality", "length", "style", "domain", "repeat_pressure"]
                and anti.get("fallback_order") == [],
                details={
                    "path": str(report_path),
                    "dataset": report_payload.get("dataset"),
                    "profile": report_payload.get("profile"),
                    "baselines": sorted(baseline_names),
                    "run_counts": sorted(run_counts),
                    "probe_protocol": protocol,
                },
            )
        )
    expected_decomposition_arms = [
        "baseline_stageA_random",
        "exact_length_style_domain",
        "exact_length_style_domain_repeat",
        "exact_length_style_domain_repeat_quality",
        "exact_length_style_domain_repeat_quality_redundancy",
    ]
    for report_path, report_payload in utility_matching_decomposition_reports:
        arm_order = report_payload.get("arm_order") or []
        summaries = report_payload.get("result_summaries") or {}
        fidelity = report_payload.get("pool_fidelity") or {}
        common_support = [
            float((fidelity.get(name) or {}).get("matched_selected_reference_ratio") or 0.0)
            for name in expected_decomposition_arms
        ]
        items.append(
            ValidationItem(
                name=f"utility_matching_decomposition_contract_{report_payload.get('dataset') or report_path.stem}",
                ok=report_payload.get("schema_version") == "utility-matching-decomposition-v1"
                and report_payload.get("selector_objective_scope") == "Stage C validation only; never selector objective"
                and arm_order == expected_decomposition_arms
                and set(summaries) == set(expected_decomposition_arms)
                and set(fidelity) == set(expected_decomposition_arms)
                and all(
                    isinstance((summaries.get(name) or {}).get("delta_nll"), (int, float))
                    and int((summaries.get(name) or {}).get("run_count") or 0) > 1
                    and bool((fidelity.get(name) or {}).get("exclude_selected"))
                    and (fidelity.get(name) or {}).get("fallback_order") == []
                    for name in expected_decomposition_arms
                )
                and all(0.0 <= value <= 1.0 for value in common_support)
                and all(
                    common_support[index] >= common_support[index + 1]
                    for index in range(len(common_support) - 1)
                ),
                details={
                    "path": str(report_path),
                    "dataset": report_payload.get("dataset"),
                    "arm_order": arm_order,
                    "matched_selected_reference_ratios": common_support,
                },
            )
        )
    for report_path, report_payload in candidate_profile_comparison_reports:
        decision = report_payload.get("decision_summary") or {}
        datasets = report_payload.get("datasets") or {}
        stage_c_gate = decision.get("stage_c_protocol_gate") or {}
        promote_candidate = bool(decision.get("promote_candidate"))
        gate_has_replicated_family = bool(stage_c_gate.get("global_replicated_default_utility_family_available"))
        items.append(
            ValidationItem(
                name=f"candidate_profile_comparison_schema_{report_path.stem}",
                ok=report_payload.get("schema_version") == "candidate-profile-comparison-v1"
                and isinstance(datasets, dict)
                and isinstance(decision, dict)
                and isinstance(decision.get("promote_candidate"), bool)
                and isinstance(decision.get("targeted_followup_candidate"), bool)
                and isinstance(stage_c_gate, dict)
                and isinstance(stage_c_gate.get("available"), bool)
                and isinstance(stage_c_gate.get("profile_matches"), bool)
                and isinstance(stage_c_gate.get("blocks_global_promotion"), bool)
                and isinstance(stage_c_gate.get("common_replicated_valid_families"), list)
                and (not promote_candidate or gate_has_replicated_family),
                details={
                    "path": str(report_path),
                    "baseline_profile": report_payload.get("baseline_profile"),
                    "candidate_profile": report_payload.get("candidate_profile"),
                    "decision_summary": decision,
                    "dataset_count": len(datasets) if isinstance(datasets, dict) else None,
                },
            )
        )
        if stage_c_protocol_decision_report and stage_c_gate.get("available"):
            protocol_global_decision = stage_c_protocol_decision_report.get("global_decision") or {}
            protocol_profile = stage_c_protocol_decision_report.get("profile")
            profile_matches = report_payload.get("candidate_profile") == protocol_profile
            items.append(
                ValidationItem(
                    name=f"candidate_profile_comparison_stage_c_gate_consistent_{report_path.stem}",
                    ok=stage_c_gate.get("profile_matches") == profile_matches
                    and stage_c_gate.get("profile_promoted") == protocol_global_decision.get("profile_promoted")
                    and stage_c_gate.get("global_default_utility_preset_available")
                    == protocol_global_decision.get("global_default_utility_preset_available")
                    and stage_c_gate.get("global_replicated_default_utility_family_available")
                    == protocol_global_decision.get("global_replicated_default_utility_family_available")
                    and stage_c_gate.get("common_valid_selected_gt_random_presets")
                    == protocol_global_decision.get("common_valid_selected_gt_random_presets")
                    and stage_c_gate.get("common_replicated_valid_families")
                    == protocol_global_decision.get("common_replicated_valid_families")
                    and stage_c_gate.get("blocks_global_promotion")
                    == (profile_matches and not bool(protocol_global_decision.get("global_replicated_default_utility_family_available"))),
                    details={
                        "path": str(report_path),
                        "candidate_profile": report_payload.get("candidate_profile"),
                        "protocol_profile": protocol_profile,
                        "stage_c_gate": stage_c_gate,
                        "protocol_global_decision": protocol_global_decision,
                    },
                )
            )
    if selector_baseline_audit:
        audit_datasets = selector_baseline_audit.get("datasets") or {}
        items.append(
            ValidationItem(
                name="selector_baseline_audit_schema",
                ok=selector_baseline_audit.get("schema_version") == "selector-baseline-audit-v1"
                and isinstance(audit_datasets, dict),
                details={
                    "path": str(SELECTOR_BASELINE_AUDIT_PATH),
                    "schema_version": selector_baseline_audit.get("schema_version"),
                    "dataset_count": len(audit_datasets) if isinstance(audit_datasets, dict) else None,
                },
            )
        )
        for dataset_name, payload in audit_datasets.items():
            comparisons = (payload or {}).get("comparisons") if isinstance(payload, dict) else None
            stage_a = (comparisons or {}).get("stageA_random") if isinstance(comparisons, dict) else None
            matched = (comparisons or {}).get("multi_matched_stageA_random") if isinstance(comparisons, dict) else None
            items.append(
                ValidationItem(
                    name=f"selector_baseline_audit_comparisons_{dataset_name}",
                    ok=isinstance(stage_a, dict)
                    and isinstance(matched, dict)
                    and isinstance((stage_a.get("verdict") or {}).get("verdict"), str)
                    and isinstance((matched.get("verdict") or {}).get("verdict"), str)
                    and isinstance(stage_a.get("top_numeric_differences"), list)
                    and isinstance(matched.get("top_numeric_differences"), list),
                    details={"stageA_random": stage_a, "multi_matched_stageA_random": matched},
                )
            )
    items.append(
        ValidationItem(
            name="run_manifest_utility_probe_path",
            ok=str(run_manifest.get("utility_probe_results_path") or "") == str(UTILITY_PROBE_RESULTS_PATH),
            details={"run_manifest_path": run_manifest.get("utility_probe_results_path"), "expected": str(UTILITY_PROBE_RESULTS_PATH)},
        )
    )
    items.append(
        ValidationItem(
            name="run_summary_utility_probe_path",
            ok=str(run_summary.get("utility_probe_results_path") or "") == str(UTILITY_PROBE_RESULTS_PATH),
            details={"run_summary_path": run_summary.get("utility_probe_results_path"), "expected": str(UTILITY_PROBE_RESULTS_PATH)},
        )
    )
    items.append(
        ValidationItem(
            name="run_manifest_core_chunk_axes",
            ok=set(run_manifest.get("core_chunk_axes") or []) == set(CORE_SELECTION_METRICS),
            details={"core_chunk_axes": run_manifest.get("core_chunk_axes")},
        )
    )
    items.append(
        ValidationItem(
            name="run_manifest_core_subset_axes",
            ok=set(run_manifest.get("core_subset_axes") or []) == set(CORE_SUBSET_METRICS),
            details={"core_subset_axes": run_manifest.get("core_subset_axes")},
        )
    )
    items.append(
        ValidationItem(
            name="run_summary_core_chunk_axes",
            ok=set(run_summary.get("core_chunk_axes") or []) == set(CORE_SELECTION_METRICS),
            details={"core_chunk_axes": run_summary.get("core_chunk_axes")},
        )
    )
    items.append(
        ValidationItem(
            name="run_summary_core_subset_axes",
            ok=set(run_summary.get("core_subset_axes") or []) == set(CORE_SUBSET_METRICS),
            details={"core_subset_axes": run_summary.get("core_subset_axes")},
        )
    )
    if "core_selection_metrics" in run_manifest:
        items.append(
            ValidationItem(
                name="run_manifest_core_selection_metrics",
                ok=set(run_manifest.get("core_selection_metrics") or []) == set(CORE_SELECTION_METRICS),
                details={"core_selection_metrics": run_manifest.get("core_selection_metrics")},
            )
        )
    if "diagnostic_metrics" in run_manifest:
        items.append(
            ValidationItem(
                name="run_manifest_diagnostic_metrics",
                ok=set(run_manifest.get("diagnostic_metrics") or []) == set(DIAGNOSTIC_METRICS),
                details={"diagnostic_metrics": run_manifest.get("diagnostic_metrics")},
            )
        )
    items.append(
        ValidationItem(
            name="utility_probe_results_schema",
            ok=str(utility_probe_results.get("schema_version") or "") == "small-lm-probe-v1",
            details={"schema_version": utility_probe_results.get("schema_version")},
        )
    )
    items.extend(_validate_profile_semantics(run_manifest))
    items.extend(_orthogonality_items(scoring_manifest))
    items.append(_validate_utility_axis_no_metric_leakage())
    items.append(_validate_selector_no_utility_surrogate())
    items.append(_validate_profile_configs_no_utility_surrogate())

    for dataset, meta in scoring_manifest.get("datasets", {}).items():
        scored_path = Path(meta["path"])
        items.extend(_validate_scored_file(scored_path))
        actual = _count_lines(scored_path)
        items.append(
            ValidationItem(
                name=f"scored_count_{dataset}",
                ok=actual == int(meta["records"]),
                details={"manifest": meta["records"], "actual": actual},
            )
        )

    for profile_name, profile in run_manifest.get("profiles", {}).items():
        profile_datasets = profile.get("datasets", {}) or {}
        profile_dataset_names = sorted(str(name) for name in profile_datasets.keys())
        for dataset, meta in profile_datasets.items():
            subset_path = Path(meta["output_path"])
            actual = _count_lines(subset_path) if subset_path.exists() else 0
            processed_records = meta.get("processed_records")
            source_records = int(meta.get("source_records") or 0)
            if processed_records is not None:
                items.append(
                    ValidationItem(
                        name=f"subset_processed_count_{profile_name}_{dataset}",
                        ok=int(processed_records) == source_records,
                        details={"manifest_processed": processed_records, "source_records": source_records},
                    )
                )
            items.append(
                ValidationItem(
                    name=f"subset_count_{profile_name}_{dataset}",
                    ok=actual == int(meta["selected_records"]),
                    details={"manifest": meta["selected_records"], "actual": actual},
                )
            )
            coverage = float(meta["subset_coverage_retention_score"])
            items.append(
                ValidationItem(
                    name=f"coverage_range_{profile_name}_{dataset}",
                    ok=0.0 <= coverage <= 1.0,
                    details={"subset_coverage_retention_score": coverage},
                )
            )
            coverage_details = meta.get("coverage_details") or {}
            source_support = coverage_details.get("source_coverage_support") or {}
            domain_support = coverage_details.get("domain_coverage_support") or {}
            style_support = coverage_details.get("style_coverage_support") or {}
            style_taxonomy_alignment = coverage_details.get("style_taxonomy_alignment") or {}
            semantic_support = coverage_details.get("semantic_coverage_support") or {}
            learning_signal_support = coverage_details.get("learning_signal_coverage_diagnostic") or {}
            items.append(
                ValidationItem(
                    name=f"coverage_source_support_present_{profile_name}_{dataset}",
                    ok=isinstance(source_support.get("distribution_similarity"), (int, float))
                    and isinstance(source_support.get("retained_bucket_ratio"), (int, float))
                    and bool(source_support.get("support_scope")),
                    details={"source_coverage_support": source_support},
                )
            )
            items.append(
                ValidationItem(
                    name=f"coverage_domain_support_present_{profile_name}_{dataset}",
                    ok=isinstance(domain_support.get("distribution_similarity"), (int, float))
                    and isinstance(domain_support.get("retained_bucket_ratio"), (int, float))
                    and bool(domain_support.get("support_scope")),
                    details={"domain_coverage_support": domain_support},
                )
            )
            items.append(
                ValidationItem(
                    name=f"coverage_style_support_present_{profile_name}_{dataset}",
                    ok=isinstance(style_support.get("distribution_similarity"), (int, float))
                    and isinstance(style_support.get("retained_bucket_ratio"), (int, float)),
                    details={"style_coverage_support": style_support},
                )
            )
            items.append(
                ValidationItem(
                    name=f"style_taxonomy_alignment_{profile_name}_{dataset}",
                    ok=style_taxonomy_alignment.get("contract")
                    == "stage_b_selected_style_equals_stage_c_full_text_recount"
                    and style_taxonomy_alignment.get("aligned") is True
                    and int(style_taxonomy_alignment.get("absolute_count_difference") or 0) == 0,
                    details={"style_taxonomy_alignment": style_taxonomy_alignment},
                )
            )
            items.append(
                ValidationItem(
                    name=f"coverage_semantic_support_present_{profile_name}_{dataset}",
                    ok=isinstance(semantic_support.get("distribution_similarity"), (int, float))
                    and isinstance(semantic_support.get("cluster_backbone_pass"), bool)
                    and semantic_support.get("support_scope") == "semantic_cluster_backbone",
                    details={"semantic_coverage_support": semantic_support},
                )
            )
            learning_gaps = learning_signal_support.get("gaps_selected_minus_baseline") or {}
            items.append(
                ValidationItem(
                    name=f"coverage_learning_signal_diagnostic_present_{profile_name}_{dataset}",
                    ok=learning_signal_support.get("policy") == "diagnostic_only_not_selector_objective"
                    and isinstance((learning_signal_support.get("selected") or {}).get("unique_bigram_ratio"), (int, float))
                    and isinstance((learning_signal_support.get("baseline") or {}).get("unique_bigram_ratio"), (int, float))
                    and isinstance(learning_gaps.get("unique_bigram_ratio"), (int, float))
                    and isinstance(learning_signal_support.get("risk_flags"), list),
                    details={"learning_signal_coverage_diagnostic": learning_signal_support},
                )
            )
            utility_score = meta.get("small_lm_probe_gain_score", meta.get("fixed_token_probe_gain_score"))
            items.append(
                ValidationItem(
                    name=f"small_lm_probe_gain_range_{profile_name}_{dataset}",
                    ok=isinstance(utility_score, (int, float)) and -1.0 <= float(utility_score) <= 1.0,
                    details={"small_lm_probe_gain_score": utility_score},
                )
            )
            stage_c = meta.get("stage_c_core_validation") or {}
            items.append(
                ValidationItem(
                    name=f"stage_c_core_validation_present_{profile_name}_{dataset}",
                    ok=isinstance(stage_c.get("passed"), bool),
                    details={"stage_c_core_validation": stage_c},
                )
            )
            for support_name, support_payload, pass_key, enforced_key in (
                ("domain", domain_support, "coverage_domain_support_pass", "coverage_domain_support_enforced"),
                ("style", style_support, "coverage_style_support_pass", "coverage_style_support_enforced"),
            ):
                thresholds = coverage_details.get(f"{support_name}_coverage_support_thresholds") or {}
                min_similarity = thresholds.get("min_distribution_similarity")
                min_retained_ratio = thresholds.get("min_retained_bucket_ratio")
                similarity = support_payload.get("distribution_similarity")
                retained_ratio = support_payload.get("retained_bucket_ratio")
                threshold_pass = (
                    isinstance(min_similarity, (int, float))
                    and isinstance(min_retained_ratio, (int, float))
                    and isinstance(similarity, (int, float))
                    and isinstance(retained_ratio, (int, float))
                    and float(similarity) >= float(min_similarity)
                    and float(retained_ratio) >= float(min_retained_ratio)
                )
                items.append(
                    ValidationItem(
                        name=f"coverage_{support_name}_support_threshold_{profile_name}_{dataset}",
                        ok=isinstance(stage_c.get(pass_key), bool)
                        and isinstance(stage_c.get(enforced_key), bool)
                        and bool(stage_c.get(pass_key)) == bool(threshold_pass),
                        details={
                            f"{support_name}_coverage_support": support_payload,
                            "thresholds": thresholds,
                            "stage_c_pass_key": stage_c.get(pass_key),
                            "stage_c_enforced_key": stage_c.get(enforced_key),
                        },
                    )
                )
            if "coverage_semantic_support_pass" in stage_c:
                items.append(
                    ValidationItem(
                        name=f"coverage_semantic_support_threshold_{profile_name}_{dataset}",
                        ok=bool(stage_c.get("coverage_semantic_support_pass")) == bool(semantic_support.get("cluster_backbone_pass")),
                        details={
                            "semantic_coverage_support": semantic_support,
                            "stage_c_pass_key": stage_c.get("coverage_semantic_support_pass"),
                        },
                    )
                )
            if "utility_mode" in stage_c:
                utility_mode = str(stage_c.get("utility_mode") or "")
                items.append(
                    ValidationItem(
                        name=f"stage_c_utility_mode_{profile_name}_{dataset}",
                        ok=utility_mode in {
                            "single_eval",
                            "in_domain_only",
                            "dual_eval_strict",
                            "in_domain_required_ood_report",
                        },
                        details={"utility_mode": utility_mode},
                    )
                )
            evaluation_mode = str(stage_c.get("evaluation_mode") or "")
            if evaluation_mode:
                items.append(
                    ValidationItem(
                        name=f"stage_c_evaluation_mode_{profile_name}_{dataset}",
                        ok=evaluation_mode in {"development", "certification"},
                        details={"evaluation_mode": evaluation_mode},
                    )
                )
            utility_details = meta.get("utility_probe_details") or {}
            utility_protocol = utility_details.get("protocol") or {}
            utility_aggregate = utility_details.get("aggregate") or {}
            items.append(
                ValidationItem(
                    name=f"utility_protocol_present_{profile_name}_{dataset}",
                    ok=isinstance(utility_protocol.get("probe_model_name"), str)
                    and isinstance(utility_protocol.get("train_token_budget"), int)
                    and isinstance(utility_protocol.get("eval_token_budget"), int)
                    and isinstance(utility_protocol.get("max_train_steps"), int)
                    and isinstance(utility_protocol.get("train_epochs"), (int, float))
                    and float(utility_protocol.get("train_epochs") or 0.0) >= 1.0
                    and isinstance(utility_protocol.get("seed_count"), int)
                    and isinstance(utility_protocol.get("holdout_bucket_count"), int)
                    and isinstance(utility_protocol.get("ood_holdout_bucket_count"), int)
                    and str(utility_protocol.get("utility_pass_statistic") or "") in {"mean", "min"},
                    details={"dataset": dataset, "protocol": utility_protocol},
                )
            )
            items.append(
                ValidationItem(
                    name=f"utility_canonical_baseline_contract_{profile_name}_{dataset}",
                    ok=utility_protocol.get("canonical_baseline") == "baseline_multi_matched_stageA_random"
                    and utility_aggregate.get("canonical_baseline") == "baseline_multi_matched_stageA_random"
                    and "baseline_stageA_random" in set(utility_protocol.get("diagnostic_baselines") or [])
                    and "baseline_stageA_random" in set(utility_aggregate.get("diagnostic_baselines") or [])
                    and "baseline_full_random" in set(utility_protocol.get("diagnostic_baselines") or [])
                    and "baseline_full_random" in set(utility_aggregate.get("diagnostic_baselines") or []),
                    details={
                        "protocol_canonical_baseline": utility_protocol.get("canonical_baseline"),
                        "aggregate_canonical_baseline": utility_aggregate.get("canonical_baseline"),
                        "protocol_diagnostic_baselines": utility_protocol.get("diagnostic_baselines"),
                        "aggregate_diagnostic_baselines": utility_aggregate.get("diagnostic_baselines"),
                    },
                )
            )
            if isinstance(utility_aggregate, dict):
                diagnostic_baselines = set(utility_aggregate.get("diagnostic_baselines") or [])
                expected_matched_baselines = {
                    "baseline_nuisance_matched_stageA_random",
                    "baseline_multi_matched_stageA_random",
                    "baseline_style_matched_stageA_random",
                    "baseline_length_matched_stageA_random",
                    "baseline_quality_band_matched_stageA_random",
                }
                optional_matched_baselines = {"baseline_anti_memorization_matched_stageA_random"}
                diagnostic_matched_baselines = expected_matched_baselines - {"baseline_multi_matched_stageA_random"}
                in_domain = utility_details.get("in_domain") or {}
                failure_analysis = utility_aggregate.get("utility_failure_analysis") or {}
                matched_pool_diagnostics = failure_analysis.get("matched_baseline_pool_diagnostics") or {}
                nuisance_diagnostics = matched_pool_diagnostics.get("baseline_nuisance_matched_stageA_random") or {}
                present_optional_matched = {
                    name
                    for name in optional_matched_baselines
                    if name in diagnostic_baselines or name in in_domain
                }
                items.append(
                    ValidationItem(
                        name=f"utility_matched_diagnostic_baselines_present_{profile_name}_{dataset}",
                        ok=diagnostic_matched_baselines.issubset(diagnostic_baselines)
                        and all(isinstance(in_domain.get(name), dict) for name in expected_matched_baselines)
                        and all(isinstance(in_domain.get(name), dict) for name in present_optional_matched)
                        and isinstance(failure_analysis.get("matched_baseline_deltas"), dict)
                        and isinstance(failure_analysis.get("failure_mode"), str),
                        details={
                            "diagnostic_baselines": sorted(diagnostic_baselines),
                            "in_domain_keys": sorted(str(name) for name in in_domain.keys()),
                            "optional_matched_baselines_present": sorted(present_optional_matched),
                            "failure_analysis": failure_analysis,
                        },
                    )
                )
                items.append(
                    ValidationItem(
                        name=f"utility_nuisance_candidate_contract_{profile_name}_{dataset}",
                        ok=nuisance_diagnostics.get("matching_policy") == "exact_length_style_domain_repeat_pressure"
                        and nuisance_diagnostics.get("matched_variables")
                        == ["length", "style", "domain", "repeat_pressure"]
                        and nuisance_diagnostics.get("excluded_selector_target_variables")
                        == ["quality", "redundancy_risk"]
                        and nuisance_diagnostics.get("fallback_order") == []
                        and (
                            isinstance(nuisance_diagnostics.get("matched_selected_reference_ratio"), (int, float))
                            or isinstance(nuisance_diagnostics.get("bucket_available"), dict)
                        )
                        and bool(nuisance_diagnostics.get("exclude_selected")),
                        details={"nuisance_candidate_diagnostics": nuisance_diagnostics},
                    )
                )
                canonical_failures = set((utility_aggregate.get("failed_by_baseline") or {}).keys())
                stress_failures = set((utility_aggregate.get("stress_failed_by_baseline") or {}).keys())
                items.append(
                    ValidationItem(
                        name=f"utility_full_random_diagnostic_only_{profile_name}_{dataset}",
                        ok="failed_vs_full_random" not in canonical_failures and "failed_vs_full_random" in stress_failures,
                        details={
                            "failed_by_baseline": utility_aggregate.get("failed_by_baseline"),
                            "stress_failed_by_baseline": utility_aggregate.get("stress_failed_by_baseline"),
                        },
                    )
                )
                baseline_control_policy = utility_aggregate.get("baseline_control_policy") or {}
                items.append(
                    ValidationItem(
                        name=f"utility_baseline_control_disjoint_{profile_name}_{dataset}",
                        ok=bool(baseline_control_policy.get("treatment_control_disjoint"))
                        and bool(baseline_control_policy.get("matched_baseline_controls_exclude_selected"))
                        and bool(baseline_control_policy.get("canonical_baseline_excludes_selected"))
                        and isinstance(baseline_control_policy.get("selected_uid_count"), int)
                        and int(baseline_control_policy.get("selected_uid_count") or 0) > 0
                        and isinstance(baseline_control_policy.get("full_random_control_uid_count"), int)
                        and int(baseline_control_policy.get("full_random_control_uid_count") or 0) > 0
                        and isinstance(baseline_control_policy.get("stageA_random_control_uid_count"), int)
                        and int(baseline_control_policy.get("stageA_random_control_uid_count") or 0) > 0
                        and baseline_control_policy.get("canonical_baseline") == "baseline_multi_matched_stageA_random"
                        and baseline_control_policy.get("canonical_matching_policy")
                        == "quality_length_style_domain_with_hierarchical_fallback"
                        and isinstance(baseline_control_policy.get("canonical_matched_pool_count"), int)
                        and int(baseline_control_policy.get("canonical_matched_pool_count") or 0) > 0,
                        details={"baseline_control_policy": baseline_control_policy},
                    )
                )
                certification_shadow = utility_aggregate.get("certification_shadow") or {}
                items.append(
                    ValidationItem(
                        name=f"utility_certification_shadow_present_{profile_name}_{dataset}",
                        ok=isinstance(certification_shadow.get("certification_ready"), bool)
                        and isinstance(certification_shadow.get("in_domain_certification_ready"), bool)
                        and isinstance(certification_shadow.get("cross_domain_certification_ready"), bool)
                        and isinstance(certification_shadow.get("domain_specific_certification_ready"), bool)
                        and isinstance(certification_shadow.get("general_purpose_certification_ready"), bool)
                        and isinstance(certification_shadow.get("strict_metric_pass"), bool)
                        and isinstance(certification_shadow.get("signal_pass"), bool)
                        and isinstance(certification_shadow.get("protocol_pass"), bool)
                        and isinstance(certification_shadow.get("probe_protocol_pass"), bool)
                        and isinstance(certification_shadow.get("evidence_tier"), str)
                        and isinstance(certification_shadow.get("blockers"), list)
                        and isinstance(certification_shadow.get("blocker_categories"), dict)
                        and isinstance(certification_shadow.get("protocol_readiness"), dict)
                        and isinstance(certification_shadow.get("in_domain_signal"), dict)
                        and isinstance(certification_shadow.get("ood_signal"), dict)
                        and isinstance(certification_shadow.get("strict_values"), dict)
                        and isinstance(certification_shadow.get("scope_snapshots"), dict)
                        and isinstance(certification_shadow.get("worst_cells"), dict)
                        and isinstance(certification_shadow.get("stability_analysis"), dict)
                        and isinstance(certification_shadow.get("step_cap_analysis"), dict),
                        details={"certification_shadow": certification_shadow},
                    )
                )
                evidence_summary = utility_aggregate.get("utility_evidence_summary") or {}
                evidence_required_number_fields = {
                    "canonical_mean_gain",
                    "canonical_in_domain_delta_nll",
                    "strict_min_gain",
                    "strict_min_relative_nll_gain",
                    "strict_min_delta_nll",
                    "strict_min_delta_nll_ci_low",
                    "max_minimum_detectable_delta_nll_95",
                    "min_effect_to_mde_ratio",
                    "min_detectable_effect_fraction",
                    "worst_in_domain_gain",
                    "worst_in_domain_delta_nll",
                }
                evidence_optional_ood_number_fields = {
                    "worst_ood_gain",
                    "worst_ood_delta_nll",
                }
                evidence_required_bool_fields = {
                    "development_pass",
                    "certification_ready",
                    "final_scope_certification_ready",
                    "in_domain_certification_ready",
                    "cross_domain_certification_ready",
                    "domain_specific_certification_ready",
                    "general_purpose_certification_ready",
                    "protocol_ready",
                    "signal_pass",
                    "in_domain_signal_pass",
                    "ood_signal_pass",
                    "in_domain_utility_axis_pass",
                    "cross_domain_utility_axis_pass",
                    "domain_specific_utility_axis_pass",
                    "general_purpose_utility_axis_pass",
                    "final_utility_axis_pass",
                }
                evidence_required_int_fields = {
                    "ood_pair_count",
                    "ood_expected_pair_count",
                    "protocol_blocker_count",
                    "signal_blocker_count",
                }
                causal_audit = utility_aggregate.get("causal_utility_audit") or {}
                items.append(
                    ValidationItem(
                        name=f"utility_evidence_summary_present_{profile_name}_{dataset}",
                        ok=isinstance(evidence_summary, dict)
                        and all(isinstance(evidence_summary.get(name), (int, float)) for name in evidence_required_number_fields)
                        and all(
                            isinstance(evidence_summary.get(name), (int, float))
                            or (
                                int(evidence_summary.get("ood_expected_pair_count") or 0) == 0
                                and evidence_summary.get(name) is None
                            )
                            for name in evidence_optional_ood_number_fields
                        )
                        and all(isinstance(evidence_summary.get(name), bool) for name in evidence_required_bool_fields)
                        and all(isinstance(evidence_summary.get(name), int) for name in evidence_required_int_fields)
                        and evidence_summary.get("evidence_tier")
                        in {
                            "development_only",
                            "in_domain_strict_signal",
                            "cross_domain_strict_signal",
                            "certification_ready",
                            "invalid_probe_evidence",
                            "not_evaluable_utility_evidence",
                            "probe_valid_token_exposure_caveat",
                            "random_baseline_gain",
                            "random_baseline_gain_with_token_exposure_caveat",
                            "matched_baseline_inconclusive",
                            "matched_baseline_gain",
                            "strict_certification_ready",
                        }
                        and evidence_summary.get("signal_status")
                        in {
                            "strict_positive",
                            "inconclusive_numerical_drift",
                            "inconclusive_below_detectable_effect",
                            "inconclusive_ci_crosses_zero",
                            "inconclusive_threshold",
                            "strict_negative",
                        }
                        and isinstance(evidence_summary.get("failure_mode"), str)
                        and evidence_summary.get("final_certification_scope") in {"domain_specific", "general_purpose"}
                        and isinstance(evidence_summary.get("signal_status_reason"), str)
                        and isinstance(evidence_summary.get("signal_interpretation"), dict)
                        and isinstance(evidence_summary.get("canonical_baseline"), str)
                        and (
                            isinstance(evidence_summary.get("worst_ood_pair"), str)
                            or (
                                int(evidence_summary.get("ood_expected_pair_count") or 0) == 0
                                and evidence_summary.get("worst_ood_pair") is None
                            )
                        )
                        and isinstance(evidence_summary.get("protocol_blockers"), list)
                        and isinstance(evidence_summary.get("signal_blockers"), list)
                        and isinstance(evidence_summary.get("certification_blockers"), list),
                        details={"utility_evidence_summary": evidence_summary},
                    )
                )
                evidence_protocol_fields_present = all(
                    isinstance(evidence_summary.get(name), dict)
                    for name in {
                        "probe_sensitivity_status",
                        "curation_benefit_status",
                        "strict_counterfactual_status",
                    }
                )
                if evidence_protocol_fields_present:
                    probe_status = evidence_summary.get("probe_sensitivity_status") or {}
                    curation_status = evidence_summary.get("curation_benefit_status") or {}
                    strict_status = evidence_summary.get("strict_counterfactual_status") or {}
                    items.append(
                        ValidationItem(
                            name=f"utility_evidence_aware_protocol_{profile_name}_{dataset}",
                            ok=(
                                probe_status.get("status")
                                in {
                                    "valid",
                                    "invalid",
                                    "not_evaluated",
                                    "probe_valid",
                                    "probe_not_evaluable",
                                    "positive_control_inconclusive_near_noise_floor",
                                    "positive_control_not_separated",
                                    "destructive_negative_inconclusive_near_noise_floor",
                                    "destructive_negative_not_separated",
                                    "probe_valid_token_exposure_confounded",
                                    "probe_valid_token_exposure_inconclusive",
                                }
                                and curation_status.get("status")
                                in {"random_baseline_gain", "random_baseline_inconclusive", "no_random_baseline_gain"}
                                and strict_status.get("status")
                                in {
                                    "strict_certification_ready",
                                    "matched_baseline_gain",
                                    "matched_baseline_inconclusive",
                                    "strict_negative",
                                }
                                and evidence_summary.get("failure_reason")
                                in {
                                    "pass",
                                    "probe_invalid",
                                    "probe_not_evaluable",
                                    "random_gain_only",
                                    "random_gain_only_with_token_exposure_caveat",
                                    "matched_inconclusive",
                                    "selected_below_stageA_random",
                                    "strict_negative",
                                }
                            ),
                            details={
                                "probe_sensitivity_status": probe_status,
                                "curation_benefit_status": curation_status,
                                "strict_counterfactual_status": strict_status,
                                "failure_reason": evidence_summary.get("failure_reason"),
                            },
                        )
                    )
                items.append(
                    ValidationItem(
                        name=f"utility_causal_audit_present_{profile_name}_{dataset}",
                        ok=isinstance(causal_audit, dict)
                        and causal_audit.get("dominant_failure_mode")
                        in {
                            "inconclusive_near_noise_floor",
                            "probe_or_training_insensitive",
                            "weaker_selected_training_signal",
                            "overfit_or_distribution_shift",
                            "positive_learning_signal",
                            "unresolved",
                        }
                        and isinstance(causal_audit.get("failure_mode_counts"), dict)
                        and isinstance(causal_audit.get("mean_eval_delta_nll"), (int, float))
                        and isinstance(causal_audit.get("mean_selected_train_audit_delta_nll"), (int, float))
                        and isinstance(causal_audit.get("mean_baseline_train_audit_delta_nll"), (int, float))
                        and isinstance(causal_audit.get("mean_selected_minus_baseline_train_audit_delta_nll"), (int, float))
                        and isinstance(causal_audit.get("probe_device_counts"), dict)
                        and isinstance(causal_audit.get("eval_batch_size_counts"), dict),
                        details={"causal_utility_audit": causal_audit},
                    )
                )
                stability_analysis = certification_shadow.get("stability_analysis") or {}
                items.append(
                    ValidationItem(
                        name=f"utility_stability_analysis_present_{profile_name}_{dataset}",
                        ok=isinstance((stability_analysis.get("combined_effective") or {}).get("noise_dominated"), bool)
                        and isinstance((stability_analysis.get("in_domain") or {}).get("available"), bool)
                        and isinstance((stability_analysis.get("ood") or {}).get("available"), bool),
                        details={"stability_analysis": stability_analysis},
                    )
                )
            profile_cfg = (run_manifest.get("profiles") or {}).get(profile_name) or {}
            stage_c_cfg = (profile_cfg.get("stage_c_validation") or {})
            cfg_eval_mode = str(stage_c_cfg.get("evaluation_mode") or "").strip().lower()
            dual_eval_required = bool(stage_c_cfg.get("enforce_ood_utility_pass")) or cfg_eval_mode == "certification"
            if dual_eval_required:
                items.append(
                    ValidationItem(
                        name=f"utility_dual_eval_enforced_{profile_name}_{dataset}",
                        ok=isinstance(utility_details.get("out_of_domain"), dict),
                        details={"has_out_of_domain": isinstance(utility_details.get("out_of_domain"), dict)},
                    )
                )
            utility_mode = str(stage_c.get("utility_mode") or "")
            if utility_mode in {"dual_eval_strict", "in_domain_required_ood_report"}:
                in_domain = utility_details.get("in_domain")
                out_of_domain = utility_details.get("out_of_domain")
                has_ood = isinstance(out_of_domain, dict)
                expected_ood_eval_datasets = sorted(name for name in profile_dataset_names if name != str(dataset))
                actual_ood_eval_datasets = sorted(str(name) for name in (out_of_domain or {}).keys()) if has_ood else []
                has_in_domain_baselines = (
                    isinstance(in_domain, dict)
                    and isinstance(in_domain.get("baseline_full_random"), dict)
                    and isinstance(in_domain.get("baseline_stageA_random"), dict)
                    and isinstance(in_domain.get("baseline_nuisance_matched_stageA_random"), dict)
                    and isinstance(in_domain.get("baseline_multi_matched_stageA_random"), dict)
                )
                has_ood_baselines = (
                    isinstance(out_of_domain, dict)
                    and actual_ood_eval_datasets == expected_ood_eval_datasets
                    and all(
                        isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_full_random"), dict)
                        and isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_stageA_random"), dict)
                        and isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_nuisance_matched_stageA_random"), dict)
                        and isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_multi_matched_stageA_random"), dict)
                        for eval_dataset in expected_ood_eval_datasets
                    )
                )
                items.append(
                    ValidationItem(
                        name=f"utility_dual_eval_details_present_{profile_name}_{dataset}",
                        ok=has_in_domain_baselines
                        and isinstance(utility_details.get("aggregate"), dict)
                        and has_ood
                        and has_ood_baselines,
                        details={
                            "dataset": dataset,
                            "has_in_domain": isinstance(in_domain, dict),
                            "has_in_domain_baselines": has_in_domain_baselines,
                            "has_out_of_domain": has_ood,
                            "has_out_of_domain_baselines": has_ood_baselines,
                            "expected_ood_eval_datasets": expected_ood_eval_datasets,
                            "actual_ood_eval_datasets": actual_ood_eval_datasets,
                            "has_aggregate": isinstance(utility_details.get("aggregate"), dict),
                            "utility_mode": utility_mode,
                        },
                    )
                )
                aggregate_pairwise_ood = (utility_details.get("aggregate") or {}).get("pairwise_ood_results")
                aggregate_ood_pair_count = (utility_details.get("aggregate") or {}).get("ood_pair_count")
                aggregate_ood_expected_pair_count = (utility_details.get("aggregate") or {}).get("ood_expected_pair_count")
                items.append(
                    ValidationItem(
                        name=f"utility_pairwise_ood_schema_{profile_name}_{dataset}",
                        ok=isinstance(aggregate_pairwise_ood, dict)
                        and sorted(str(name) for name in aggregate_pairwise_ood.keys()) == expected_ood_eval_datasets
                        and isinstance(aggregate_ood_pair_count, int)
                        and aggregate_ood_pair_count == len(expected_ood_eval_datasets)
                        and isinstance(aggregate_ood_expected_pair_count, int)
                        and aggregate_ood_expected_pair_count == len(expected_ood_eval_datasets),
                        details={
                            "dataset": dataset,
                            "expected_ood_pair_count": len(expected_ood_eval_datasets),
                            "aggregate_ood_pair_count": aggregate_ood_pair_count,
                            "aggregate_ood_expected_pair_count": aggregate_ood_expected_pair_count,
                            "aggregate_eval_datasets": sorted(str(name) for name in (aggregate_pairwise_ood or {}).keys()) if isinstance(aggregate_pairwise_ood, dict) else None,
                        },
                    )
                )
                if has_in_domain_baselines:
                    token_fields_ok = True
                    paired_probe_ok = True
                    token_field_details = {}
                    for baseline_name, baseline_payload in in_domain.items():
                        selected_tokens = baseline_payload.get("selected_train_tokens_mean")
                        baseline_tokens = baseline_payload.get("baseline_train_tokens_mean")
                        selected_steps = baseline_payload.get("selected_effective_train_steps_mean")
                        baseline_steps = baseline_payload.get("baseline_effective_train_steps_mean")
                        selected_seen_tokens = baseline_payload.get("selected_estimated_seen_train_tokens_mean")
                        baseline_seen_tokens = baseline_payload.get("baseline_estimated_seen_train_tokens_mean")
                        selected_exposure = baseline_payload.get("selected_train_token_exposure_ratio_mean")
                        baseline_exposure = baseline_payload.get("baseline_train_token_exposure_ratio_mean")
                        selected_target_exposure = baseline_payload.get("selected_target_train_exposure_ratio_mean")
                        baseline_target_exposure = baseline_payload.get("baseline_target_train_exposure_ratio_mean")
                        train_epochs = baseline_payload.get("train_epochs_mean")
                        paired_bootstrap = baseline_payload.get("paired_bootstrap")
                        mde_delta = baseline_payload.get("minimum_detectable_delta_nll_95_max")
                        effect_to_mde = baseline_payload.get("effect_to_mde_ratio_min")
                        detectable_fraction = baseline_payload.get("detectable_effect_fraction")
                        token_field_details[baseline_name] = {
                            "selected_train_tokens_mean": selected_tokens,
                            "baseline_train_tokens_mean": baseline_tokens,
                            "selected_effective_train_steps_mean": selected_steps,
                            "baseline_effective_train_steps_mean": baseline_steps,
                            "selected_estimated_seen_train_tokens_mean": selected_seen_tokens,
                            "baseline_estimated_seen_train_tokens_mean": baseline_seen_tokens,
                            "selected_train_token_exposure_ratio_mean": selected_exposure,
                            "baseline_train_token_exposure_ratio_mean": baseline_exposure,
                            "selected_target_train_exposure_ratio_mean": selected_target_exposure,
                            "baseline_target_train_exposure_ratio_mean": baseline_target_exposure,
                            "train_epochs_mean": train_epochs,
                            "paired_bootstrap": paired_bootstrap,
                            "minimum_detectable_delta_nll_95_max": mde_delta,
                            "effect_to_mde_ratio_min": effect_to_mde,
                            "detectable_effect_fraction": detectable_fraction,
                        }
                        token_fields_ok = token_fields_ok and isinstance(selected_tokens, int) and selected_tokens > 0
                        token_fields_ok = token_fields_ok and isinstance(baseline_tokens, int) and baseline_tokens > 0
                        token_fields_ok = token_fields_ok and isinstance(selected_steps, int) and selected_steps > 0
                        token_fields_ok = token_fields_ok and isinstance(baseline_steps, int) and baseline_steps > 0
                        token_fields_ok = token_fields_ok and isinstance(selected_seen_tokens, int) and selected_seen_tokens > 0
                        token_fields_ok = token_fields_ok and isinstance(baseline_seen_tokens, int) and baseline_seen_tokens > 0
                        token_fields_ok = token_fields_ok and isinstance(selected_exposure, (int, float)) and float(selected_exposure) > 0.0
                        token_fields_ok = token_fields_ok and isinstance(baseline_exposure, (int, float)) and float(baseline_exposure) > 0.0
                        token_fields_ok = token_fields_ok and isinstance(selected_target_exposure, (int, float)) and float(selected_target_exposure) >= 1.0
                        token_fields_ok = token_fields_ok and isinstance(baseline_target_exposure, (int, float)) and float(baseline_target_exposure) >= 1.0
                        token_fields_ok = token_fields_ok and isinstance(train_epochs, (int, float)) and float(train_epochs) >= 1.0
                        paired_probe_ok = paired_probe_ok and bool(paired_bootstrap)
                        paired_probe_ok = paired_probe_ok and isinstance(mde_delta, (int, float)) and float(mde_delta) >= 0.0
                        paired_probe_ok = paired_probe_ok and isinstance(effect_to_mde, (int, float))
                        paired_probe_ok = paired_probe_ok and isinstance(detectable_fraction, (int, float)) and 0.0 <= float(detectable_fraction) <= 1.0
                    items.append(
                        ValidationItem(
                            name=f"utility_train_token_fields_present_{profile_name}_{dataset}",
                            ok=bool(token_fields_ok),
                            details=token_field_details,
                        )
                    )
                    items.append(
                        ValidationItem(
                            name=f"utility_paired_mde_fields_present_{profile_name}_{dataset}",
                            ok=bool(paired_probe_ok),
                            details=token_field_details,
                        )
                    )
            cluster_backbone_audit = meta.get("cluster_backbone_audit") or {}
            items.append(
                ValidationItem(
                    name=f"coverage_cluster_backbone_present_{profile_name}_{dataset}",
                    ok=isinstance(cluster_backbone_audit.get("passed"), bool)
                    and isinstance(cluster_backbone_audit.get("lexical_separation_pass"), bool)
                    and isinstance(cluster_backbone_audit.get("within_gt_between_fraction"), (int, float))
                    and isinstance(cluster_backbone_audit.get("within_pair_count"), int)
                    and isinstance(cluster_backbone_audit.get("between_pair_count"), int)
                    and cluster_backbone_audit.get("anchor_purity_role") == "diagnostic_only",
                    details={"cluster_backbone_audit": cluster_backbone_audit},
                )
            )

    _validate_slm_update_experiment_manifests(items)
    if historical_evidence_enabled:
        _validate_temporal_code_stage_b(items)

    if DASHBOARD_PATH.exists():
        if "Training Data Evaluation Dashboard" not in DASHBOARD_PATH.read_text(encoding="utf-8", errors="replace"):
            items.append(ValidationItem(name="dashboard_title", ok=False, details={"path": str(DASHBOARD_PATH)}))
        else:
            items.append(ValidationItem(name="dashboard_title", ok=True, details={"path": str(DASHBOARD_PATH)}))
    else:
        items.append(ValidationItem(name="dashboard_title", ok=False, details={"path": str(DASHBOARD_PATH), "reason": "dashboard missing"}))

    if PROPERTY_BENCHMARK_DIR.exists():
        for report_path in sorted(PROPERTY_BENCHMARK_DIR.glob("*_property_benchmark_report.json")):
            report = load_json(report_path)
            dataset = str(report.get("dataset") or report_path.stem.replace("_property_benchmark_report", ""))
            audits = report.get("diagnostic_audits") or {}
            validity_audit = audits.get("validity_behavior") or {}
            quality_audit = audits.get("quality_domain_shift") or {}
            redundancy_audit = audits.get("redundancy_behavior") or {}
            items.append(
                ValidationItem(
                    name=f"property_benchmark_validity_audit_present_{dataset}",
                    ok=isinstance(validity_audit.get("violated_rule_counts"), dict)
                    and isinstance((validity_audit.get("repetition_only_failures") or {}).get("count"), int)
                    and isinstance(validity_audit.get("decision_scope_counts"), dict)
                    and isinstance(validity_audit.get("hard_warning_boundary"), dict),
                    details={"path": str(report_path), "validity_behavior": validity_audit},
                )
            )
            items.append(
                ValidationItem(
                    name=f"property_benchmark_quality_audit_present_{dataset}",
                    ok=isinstance(quality_audit.get("by_style_bucket"), dict)
                    and isinstance(quality_audit.get("by_domain_bucket_top"), dict)
                    and isinstance(quality_audit.get("by_length_bucket"), dict)
                    and isinstance(quality_audit.get("valid_but_low_quality"), dict),
                    details={"path": str(report_path), "quality_domain_shift": quality_audit},
                )
            )
            items.append(
                ValidationItem(
                    name=f"property_benchmark_redundancy_audit_present_{dataset}",
                    ok=isinstance(redundancy_audit.get("by_style_bucket"), dict)
                    and isinstance(redundancy_audit.get("intra_chunk_repetition"), dict),
                    details={"path": str(report_path), "redundancy_behavior": redundancy_audit},
                )
            )
            items.append(
                ValidationItem(
                    name=f"property_benchmark_assertion_summary_{dataset}",
                    ok=int((report.get("summary") or {}).get("supported_assertions") or 0)
                    == sum(1 for a in (report.get("assertions") or []) if a.get("supported")),
                    details={"path": str(report_path), "summary": report.get("summary")},
                )
            )

    return items


def main(write_report: Path | None = VALIDATION_REPORT_PATH, scope: str = "full") -> int:
    items = validate_outputs(scope=scope)
    passed = [x for x in items if x.ok]
    failed = [x for x in items if not x.ok]
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "summary": {
            "total": len(items),
            "passed": len(passed),
            "failed": len(failed),
        },
        "items": [x.__dict__ for x in items],
        "results": [x.__dict__ for x in items],
    }
    if write_report is not None:
        save_json(write_report, report)
        build_metric_maturity_snapshot(validation_report_path=write_report)
    print("Validation summary:")
    print(f"  total: {len(items)}")
    print(f"  pass:  {len(passed)}")
    print(f"  fail:  {len(failed)}")
    if failed:
        for item in failed[:10]:
            print(f"  - {item.name}: {item.details}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
