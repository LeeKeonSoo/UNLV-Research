#!/usr/bin/env python3
"""Summary and manifest writers for the generic data evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from data_eval_common import (
    CORE_SUBSET_METRICS,
    CORE_SELECTION_METRICS,
    DIAGNOSTIC_METRICS,
    METRIC_SPEC_PATH,
    RUN_MANIFEST_PATH,
    RUN_SUMMARY_PATH,
    SCHEMA_VERSION,
    fingerprint_files,
    save_json,
)


def write_run_reports(
    profiles_path: Path,
    index_db_path: Path,
    scoring_manifest_path: Path,
    scoring_manifest: Dict[str, Any],
    profile_summaries: Dict[str, Any],
    utility_probe_results_path: Path | None = None,
) -> Dict[str, Any]:
    run_manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "core_chunk_axes": list(CORE_SELECTION_METRICS),
        "core_subset_axes": list(CORE_SUBSET_METRICS),
        "core_selection_metrics": list(CORE_SELECTION_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "metric_spec_path": str(METRIC_SPEC_PATH),
        "metric_spec_fingerprint": fingerprint_files([METRIC_SPEC_PATH]),
        "profiles_path": str(profiles_path),
        "index_db_path": str(index_db_path),
        "scoring_manifest_path": str(scoring_manifest_path),
        "utility_probe_results_path": str(utility_probe_results_path) if utility_probe_results_path else None,
        "profiles": profile_summaries,
        "datasets": scoring_manifest["datasets"],
    }

    run_summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "core_chunk_axes": list(CORE_SELECTION_METRICS),
        "core_subset_axes": list(CORE_SUBSET_METRICS),
        "core_selection_metrics": list(CORE_SELECTION_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "metric_spec_path": str(METRIC_SPEC_PATH),
        "metric_spec_fingerprint": fingerprint_files([METRIC_SPEC_PATH]),
        "utility_probe_results_path": str(utility_probe_results_path) if utility_probe_results_path else None,
        "profiles": {},
    }
    for profile_name, profile in profile_summaries.items():
        profile_summary = {
            dataset: {
                "selected_records": meta["selected_records"],
                "full_curated_pool_records": meta.get("full_curated_pool_records"),
                "budget_not_selected_records": meta.get("budget_not_selected_records"),
                "stage_a_rejected_records": meta.get("stage_a_rejected_records"),
                "stage_b_selection_mode": meta.get("stage_b_selection_mode"),
                "disposition_summary": meta.get("disposition_summary") or {},
                "disposition_invariants": meta.get("disposition_invariants") or {},
                "curated_pool_reference": meta.get("curated_pool_reference") or {},
                "processed_records": meta.get("processed_records"),
                "source_records": meta.get("source_records"),
                "selection_ratio": meta["selection_ratio"],
                "subset_coverage_retention_score": meta["subset_coverage_retention_score"],
                "small_lm_probe_gain_score": meta.get("small_lm_probe_gain_score"),
                "fixed_token_probe_gain_score": meta.get("fixed_token_probe_gain_score"),
                "coverage_details": meta.get("coverage_details") or {},
                "cluster_backbone_audit": meta.get("cluster_backbone_audit") or {},
                "utility_probe_details": meta.get("utility_probe_details") or {},
                "stage_b_rank_effective": meta.get("stage_b_rank_effective") or {},
                "selector_diagnostics": meta.get("selector_diagnostics") or {},
                "coverage_strategy_details": meta.get("coverage_strategy_details") or {},
                "output_path": meta.get("output_path"),
                "core_axes": {
                    "coverage": {
                        "score": meta["subset_coverage_retention_score"],
                        "details": {
                            **(meta.get("coverage_details") or {}),
                            "cluster_backbone_audit": meta.get("cluster_backbone_audit") or {},
                        },
                        "pass": bool((meta.get("stage_c_core_validation") or {}).get("coverage_pass")),
                    },
                    "utility": {
                        "score": meta.get("small_lm_probe_gain_score"),
                        "details": (meta.get("utility_probe_details") or {}).get("aggregate") or {},
                        "pass": bool((meta.get("stage_c_core_validation") or {}).get("utility_axis_pass")),
                    },
                },
                "stage_c_core_validation": {
                    **(meta.get("stage_c_core_validation") or {}),
                    "utility_pass": bool((meta.get("stage_c_core_validation") or {}).get("utility_axis_pass")),
                },
                "stage_c_fail_reasons": {
                    "coverage": [] if bool((meta.get("stage_c_core_validation") or {}).get("coverage_pass")) else [
                        k
                        for k in (
                            "coverage_score_pass",
                            "coverage_tail_retention_pass",
                            "coverage_tail_count_pass",
                            "coverage_backbone_pass",
                            "coverage_semantic_support_pass",
                            "coverage_domain_support_pass",
                            "coverage_style_support_pass",
                        )
                        if not bool((meta.get("stage_c_core_validation") or {}).get(k))
                        and (
                            (
                                k != "coverage_backbone_pass"
                                or bool((meta.get("stage_c_core_validation") or {}).get("coverage_backbone_enforced"))
                            )
                            and (
                                k != "coverage_semantic_support_pass"
                                or bool((meta.get("stage_c_core_validation") or {}).get("coverage_semantic_support_enforced"))
                            )
                            and (
                                k != "coverage_domain_support_pass"
                                or bool((meta.get("stage_c_core_validation") or {}).get("coverage_domain_support_enforced"))
                            )
                            and (
                                k != "coverage_style_support_pass"
                                or bool((meta.get("stage_c_core_validation") or {}).get("coverage_style_support_enforced"))
                            )
                        )
                    ],
                    "utility": [] if bool((meta.get("stage_c_core_validation") or {}).get("utility_axis_pass")) else [
                        k
                        for k in (
                            "utility_score_pass",
                            "utility_relative_gain_pass",
                            "utility_delta_nll_pass",
                            "utility_ci_pass",
                        )
                        if not bool((meta.get("stage_c_core_validation") or {}).get(k))
                    ]
                    + [
                        baseline_key
                        for baseline_key, failed in (
                            (meta.get("stage_c_core_validation") or {}).get("utility_failures_by_baseline") or {}
                        ).items()
                        if bool(failed)
                    ],
                },
            }
            for dataset, meta in profile["datasets"].items()
        }
        profile_summary["_evaluation_mode"] = str(
            (profile.get("stage_c_validation") or {}).get("evaluation_mode") or "development"
        )
        run_summary["profiles"][profile_name] = profile_summary

    save_json(RUN_MANIFEST_PATH, run_manifest)
    save_json(RUN_SUMMARY_PATH, run_summary)
    return run_manifest
