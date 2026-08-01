#!/usr/bin/env python3
"""Build a metric maturity snapshot from the latest pipeline outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from data_eval_common import (
    CORE_SUBSET_METRICS,
    CORE_SELECTION_METRICS,
    DIAGNOSTIC_METRICS,
    METRIC_MATURITY_SNAPSHOT_PATH,
    METRIC_SPEC_PATH,
    RUN_SUMMARY_PATH,
    SCORED_DIR,
    VALIDATION_OUTPUT_DIR,
    fingerprint_files,
    load_json,
    save_json,
)


SCORING_MANIFEST_PATH = SCORED_DIR / "scoring_manifest.json"
VALIDATION_REPORT_PATH = VALIDATION_OUTPUT_DIR / "full_validation_report.json"
PROPERTY_BENCHMARK_DIR = VALIDATION_OUTPUT_DIR / "property_benchmarks"
TRACKED_METRICS = (
    "structural_validity_gate",
    "structural_validity_score",
    "reference_quality_score",
    "exact_duplicate_indicator",
    "shingle_near_duplicate_indicator",
    "shingle_near_duplicate_risk_score",
    "subset_coverage_retention_score",
    "small_lm_probe_gain_score",
    "fixed_token_probe_gain_score",
    "explanatory_quality_proxy",
    "tail_cluster_rarity_proxy",
    "predictive_utility_proxy",
)
MATURITY_POLICY: Dict[str, Dict[str, str]] = {
    "structural_validity_gate": {
        "maturity": "High",
        "action": "Freeze",
        "method": "Rule-based structural gate with explicit violated-rule reporting",
        "notes": "Canonical Stage A gate for structural usability; separates pass/fail from the soft validity score.",
    },
    "structural_validity_score": {
        "maturity": "Mid",
        "action": "Validate",
        "method": "Diagnostic structural usability score with hard-failure and warning audit fields",
        "notes": "Diagnostic support only; canonical Validity pass/fail is structural_validity_gate.",
    },
    "reference_quality_score": {
        "maturity": "High",
        "action": "Freeze",
        "method": "Reference-trained selection-value evidence with style/length-normalized calibration",
        "notes": "Canonical Stage B budget-allocation evidence. Quality is a legacy alias; the score has no hard-reject authority and Stage-A-pass records remain in the full curated pool.",
    },
    "exact_duplicate_indicator": {
        "maturity": "High",
        "action": "Freeze",
        "method": "Hash-based exact duplicate detection",
        "notes": "Most direct metric in the framework; semantics and implementation are closely aligned.",
    },
    "shingle_near_duplicate_indicator": {
        "maturity": "Mid",
        "action": "Validate",
        "method": "Adaptive SimHash shortlist plus verified token 3-gram Jaccard overlap",
        "notes": "Much closer to literature-style fuzzy dedup than the previous density proxy, but still needs broader benchmarking.",
    },
    "shingle_near_duplicate_risk_score": {
        "maturity": "Mid",
        "action": "Validate",
        "method": "Continuous harmful-redundancy risk from verified overlap, prefix pressure, repetition pressure, and useful-recurrence relief",
        "notes": "Separates harmful duplicate burden from useful recurrence in definitions, examples, exercises, and technical references; still needs broader dedup benchmark calibration.",
    },
    "explanatory_quality_proxy": {
        "maturity": "Low",
        "action": "Replace",
        "method": "HashingVectorizer prototype similarity plus handcrafted explanatory signals",
        "notes": "Diagnostic only; kept for comparison against the newer reference-trained quality classifier.",
    },
    "tail_cluster_rarity_proxy": {
        "maturity": "Low",
        "action": "Replace",
        "method": "MiniBatchKMeans cluster rarity proxy",
        "notes": "Rarity is not equivalent to true coverage; keep it diagnostic and use subset-level coverage as the authoritative evaluator.",
    },
    "subset_coverage_retention_score": {
        "maturity": "High",
        "action": "Freeze",
        "method": "Subset-level source/style/semantic retention with cluster-backbone audit",
        "notes": "Authoritative coverage axis in Stage C. Explicit domain metadata is distinguished from source-bucket fallback so source coverage is not over-claimed as semantic domain coverage.",
    },
    "small_lm_probe_gain_score": {
        "maturity": "High",
        "action": "Freeze",
        "method": "Tiny causal LM fixed-budget finetune with held-out NLL uplift",
        "notes": "Authoritative utility axis in Stage C with canonical Stage-A baseline and all-pairwise OOD gating.",
    },
    "fixed_token_probe_gain_score": {
        "maturity": "Low",
        "action": "Replace",
        "method": "Deprecated hybrid n-gram held-out loss probe",
        "notes": "Retained as compatibility output only; no longer canonical for utility.",
    },
    "predictive_utility_proxy": {
        "maturity": "Low",
        "action": "Monitor",
        "method": "Heuristic utility proxy with optional gated predictor",
        "notes": "Diagnostic/development-only surrogate. It is excluded from canonical selection, Stage-C pass/fail, and certification readiness.",
    },
}

METRIC_MARGIN_THRESHOLDS: Dict[str, float] = {
    "structural_validity_gate": 0.5,
    "structural_validity_score": 0.02,
    "reference_quality_score": 0.10,
    "exact_duplicate_indicator": 0.95,
    "shingle_near_duplicate_indicator": 0.005,
    "shingle_near_duplicate_risk_score": 0.05,
    "small_lm_probe_gain_score": 0.001,
    "fixed_token_probe_gain_score": 0.001,
    "explanatory_quality_proxy": 0.05,
    "tail_cluster_rarity_proxy": 0.10,
    "predictive_utility_proxy": 0.10,
}


def _role_for_metric(metric_name: str) -> str:
    if metric_name in CORE_SUBSET_METRICS:
        return "Subset Evaluator"
    if metric_name in CORE_SELECTION_METRICS:
        return "Canonical Core"
    return "Diagnostic"


def _load_property_reports() -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    if not PROPERTY_BENCHMARK_DIR.exists():
        return reports
    for path in sorted(PROPERTY_BENCHMARK_DIR.glob("*_property_benchmark_report.json")):
        payload = load_json(path)
        dataset_name = str(payload.get("dataset") or path.stem.replace("_property_benchmark_report", ""))
        reports[dataset_name] = payload
    return reports


def _property_assertion_validation(metric_name: str, property_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    matched = 0
    passed = 0
    failed = 0
    datasets: Dict[str, str] = {}
    for dataset_name, report in property_reports.items():
        assertions = [a for a in report.get("assertions", []) if a.get("metric") == metric_name]
        if not assertions:
            datasets[dataset_name] = "Mixed"
            continue
        matched += len(assertions)
        if all(bool(a.get("passed")) for a in assertions):
            passed += len(assertions)
            datasets[dataset_name] = "Pass"
        else:
            failed_here = sum(1 for a in assertions if not bool(a.get("passed")))
            passed_here = len(assertions) - failed_here
            passed += passed_here
            failed += failed_here
            datasets[dataset_name] = "Fail"
    if matched == 0:
        label = "Mixed"
    elif failed == 0:
        label = "Pass"
    elif passed == 0:
        label = "Fail"
    else:
        label = "Mixed"
    return {
        "label": label,
        "matched_assertions": matched,
        "passed_assertions": passed,
        "failed_assertions": failed,
        "datasets": datasets,
    }


def _benchmark_audit_evidence(metric_name: str, property_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    audits: Dict[str, Any] = {}
    for dataset_name, report in property_reports.items():
        diagnostic_audits = report.get("diagnostic_audits") or {}
        if metric_name in {"structural_validity_gate", "structural_validity_score"}:
            audits[dataset_name] = diagnostic_audits.get("validity_behavior") or {}
        elif metric_name == "reference_quality_score":
            audits[dataset_name] = diagnostic_audits.get("quality_domain_shift") or {}
        elif metric_name in {
            "exact_duplicate_indicator",
            "shingle_near_duplicate_indicator",
            "shingle_near_duplicate_risk_score",
        }:
            audits[dataset_name] = diagnostic_audits.get("redundancy_behavior") or {}
    return audits


def _metric_assertion_details(metric_name: str, property_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, Dict[str, Any]] = {}
    margins: List[float] = []
    supported = 0
    passed = 0
    for dataset_name, report in property_reports.items():
        assertions = [a for a in report.get("assertions", []) if a.get("metric") == metric_name and a.get("supported")]
        if not assertions:
            by_dataset[dataset_name] = {
                "supported_assertions": 0,
                "passed_assertions": 0,
                "all_passed": False,
                "max_margin": None,
                "mean_margin": None,
            }
            continue
        local_margins = [float(a.get("margin") or 0.0) for a in assertions if a.get("margin") is not None]
        local_passed = sum(1 for a in assertions if bool(a.get("passed")))
        supported += len(assertions)
        passed += local_passed
        margins.extend(local_margins)
        by_dataset[dataset_name] = {
            "supported_assertions": len(assertions),
            "passed_assertions": local_passed,
            "all_passed": local_passed == len(assertions),
            "max_margin": max(local_margins) if local_margins else None,
            "mean_margin": (sum(local_margins) / len(local_margins)) if local_margins else None,
        }
    return {
        "supported_assertions": supported,
        "passed_assertions": passed,
        "all_passed": (supported > 0 and passed == supported),
        "max_margin": max(margins) if margins else None,
        "mean_margin": (sum(margins) / len(margins)) if margins else None,
        "datasets": by_dataset,
    }


def _profile_fail_reasons_by_metric(property_reports: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for report in property_reports.values():
        profile_diag = report.get("profile_gate_diagnostics") or {}
        for profile_stats in profile_diag.values():
            fail_reasons = profile_stats.get("fail_reasons") or {}
            for reason, count in fail_reasons.items():
                if reason == "stage_b:near_duplicate_risk_ceiling":
                    out["shingle_near_duplicate_risk_score"] = out.get("shingle_near_duplicate_risk_score", 0) + int(count or 0)
                    continue
                if reason == "stage_b:selection_threshold":
                    out["reference_quality_score"] = out.get("reference_quality_score", 0) + int(count or 0)
                    continue
                if ":" not in reason:
                    continue
                _, metric = reason.split(":", 1)
                out[metric] = out.get(metric, 0) + int(count or 0)
    return out


def _stability_gate(metric_name: str, means: Dict[str, float | None]) -> Tuple[str, str]:
    numeric_means = {
        str(dataset): float(value)
        for dataset, value in means.items()
        if isinstance(value, (int, float))
    }
    if len(numeric_means) < 2:
        return "Mixed", "fewer than two dataset means available"
    if metric_name in {"exact_duplicate_indicator", "shingle_near_duplicate_indicator", "shingle_near_duplicate_risk_score"}:
        # Sparse-by-design metrics should stay low but non-degenerate across available datasets.
        if all(value > 0.0 for value in numeric_means.values()):
            return "Pass", f"sparse redundancy signals present on {len(numeric_means)} datasets"
        return "Fail", "redundancy signal missing on at least one dataset"
    # For continuous quality/validity-style metrics, require reasonable cross-dataset agreement.
    spread = max(numeric_means.values()) - min(numeric_means.values())
    if spread <= 0.25:
        return "Pass", f"cross-dataset mean spread={spread:.6f}"
    return "Fail", f"cross-dataset mean spread too large ({spread:.6f})"


def _identity_gate(metric_name: str, assertion_details: Dict[str, Any], validation: Dict[str, Any] | None = None) -> Tuple[str, str]:
    if metric_name == "subset_coverage_retention_score":
        if validation and validation.get("label") == "Pass":
            return "Pass", "subset coverage identity is validated in development mode"
        return "Fail", "subset coverage support/backbone validation is incomplete"
    if metric_name in {"small_lm_probe_gain_score", "fixed_token_probe_gain_score"}:
        if validation and validation.get("label") == "Pass":
            return "Pass", "subset utility identity is validated in development mode"
        return "Fail", "subset utility protocol validation is incomplete"
    supported = int(assertion_details.get("supported_assertions") or 0)
    if supported == 0:
        return "Mixed", "no supported assertions for this metric"
    if not bool(assertion_details.get("all_passed")):
        return "Fail", "one or more supported assertions failed"
    threshold = METRIC_MARGIN_THRESHOLDS.get(metric_name, 0.0)
    max_margin = assertion_details.get("max_margin")
    if max_margin is None:
        return "Mixed", "assertions passed but margin unavailable"
    if float(max_margin) < float(threshold):
        return "Fail", f"max margin {float(max_margin):.6f} below threshold {float(threshold):.6f}"
    return "Pass", f"max margin {float(max_margin):.6f} >= threshold {float(threshold):.6f}"


def _impact_gate(
    metric_name: str,
    property_reports: Dict[str, Dict[str, Any]],
    fail_reason_counts: Dict[str, int],
) -> Tuple[str, str]:
    if metric_name == "subset_coverage_retention_score":
        return "Pass", "subset coverage evaluator is directly used in profile-level validation"
    if metric_name in {"small_lm_probe_gain_score", "fixed_token_probe_gain_score"}:
        return "Pass", "fixed-token probe evaluator is directly used in profile-level validation"
    if metric_name not in CORE_SELECTION_METRICS:
        return "Mixed", "diagnostic/outcome metric (not a direct selector)"
    count = int(fail_reason_counts.get(metric_name) or 0)
    if count > 0:
        return "Pass", f"profile gates recorded {count} failures on this metric"
    if metric_name == "shingle_near_duplicate_risk_score":
        # Stage-B redundancy ranking uses risk score even when hard gates are sparse.
        has_profile_diag = any(bool((r.get("profile_gate_diagnostics") or {})) for r in property_reports.values())
        if has_profile_diag:
            return "Pass", "risk score participates in stage-b redundancy ranking"
    return "Fail", "no measurable selection impact in current profile diagnostics"


def _completion_level(gates: Dict[str, str]) -> str:
    labels = [("Fail" if v == "Mixed" else v) for v in gates.values() if v in {"Pass", "Fail", "Mixed"}]
    if not labels:
        return "Low"
    passed = sum(1 for v in labels if v == "Pass")
    ratio = passed / len(labels)
    if ratio >= 1.0:
        return "High"
    if ratio >= 0.5:
        return "Mid"
    return "Low"


def _summary_profile_datasets(run_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Return the active subset profile's dataset map from old or new summary layouts."""
    profiles = run_summary.get("profiles", {}) or {}
    candidate = profiles.get("canonical")
    if candidate is None:
        for _, payload in profiles.items():
            if isinstance(payload, dict):
                candidate = payload
                break
    if not isinstance(candidate, dict):
        return {}
    datasets = candidate.get("datasets")
    if isinstance(datasets, dict):
        return datasets
    return {
        str(dataset_name): meta
        for dataset_name, meta in candidate.items()
        if isinstance(meta, dict) and not str(dataset_name).startswith("_")
    }



def _subset_metric_validation(
    validation_report: Dict[str, Any],
    metric_name: str,
    run_summary: Dict[str, Any],
) -> Dict[str, Any]:
    items = validation_report.get("items") or validation_report.get("results", [])
    if metric_name == "subset_coverage_retention_score":
        prefixes = (
            "coverage_cluster_backbone_present_",
            "coverage_source_support_present_",
            "coverage_domain_support_present_",
            "coverage_style_support_present_",
            "coverage_semantic_support_present_",
        )
    else:
        prefixes = (
            "utility_protocol_present_",
            "utility_dual_eval_details_present_",
        )
    checks = [r for r in items if any(str(r.get("name") or "").startswith(prefix) for prefix in prefixes)]
    if not checks:
        return {
            "label": "Mixed",
            "matched_assertions": 0,
            "passed_assertions": 0,
            "failed_assertions": 0,
            "datasets": {},
        }
    datasets = {}
    passed = 0
    failed = 0
    by_dataset: Dict[str, List[bool]] = {}
    for check in checks:
        details = check.get("details") or {}
        dataset = str(details.get("dataset") or "")
        if not dataset:
            name = str(check.get("name", ""))
            for prefix in prefixes:
                if name.startswith(prefix):
                    dataset = name.replace(prefix, "", 1).split("_", 1)[-1]
                    break
        by_dataset.setdefault(dataset, []).append(bool(check.get("ok")))
    for dataset, ok_values in by_dataset.items():
        ok = all(ok_values)
        datasets[dataset] = "Pass" if ok else "Fail"
        if ok:
            passed += 1
        else:
            failed += 1
    label = "Pass" if failed == 0 and by_dataset else ("Fail" if passed == 0 and by_dataset else "Mixed")
    payload = {
        "label": label,
        "matched_assertions": len(checks),
        "passed_assertions": passed,
        "failed_assertions": failed,
        "datasets": datasets,
    }
    profile = _summary_profile_datasets(run_summary)
    readiness_by_dataset: Dict[str, Any] = {}
    certification_passes: List[bool] = []
    for dataset, meta in profile.items():
        if str(dataset).startswith("_") or not isinstance(meta, dict):
            continue
        stage_c = meta.get("stage_c_core_validation") or {}
        if metric_name == "subset_coverage_retention_score":
            audit = meta.get("cluster_backbone_audit") or {}
            coverage_details = meta.get("coverage_details") or {}
            ready = bool(stage_c.get("coverage_pass")) and bool(audit.get("passed"))
            source_support = coverage_details.get("source_coverage_support") or {}
            domain_support = coverage_details.get("domain_coverage_support") or {}
            style_support = coverage_details.get("style_coverage_support") or {}
            semantic_support = coverage_details.get("semantic_coverage_support") or {}
            readiness_by_dataset[str(dataset)] = {
                "certification_ready": ready,
                "development_pass": bool(stage_c.get("coverage_pass")),
                "backbone_pass": bool(audit.get("passed")),
                "source_support_scope": source_support.get("support_scope"),
                "source_support_pass": bool(
                    isinstance(source_support.get("distribution_similarity"), (int, float))
                    and isinstance(source_support.get("retained_bucket_ratio"), (int, float))
                ),
                "source_coverage_support": source_support,
                "domain_support_pass": bool(stage_c.get("coverage_domain_support_pass")),
                "domain_support_enforced": bool(stage_c.get("coverage_domain_support_enforced")),
                "domain_support_scope": domain_support.get("support_scope"),
                "domain_coverage_support": domain_support,
                "style_support_pass": bool(stage_c.get("coverage_style_support_pass")),
                "style_support_enforced": bool(stage_c.get("coverage_style_support_enforced")),
                "style_coverage_support": style_support,
                "semantic_support_pass": bool(stage_c.get("coverage_semantic_support_pass")),
                "semantic_support_enforced": bool(stage_c.get("coverage_semantic_support_enforced")),
                "semantic_coverage_support": semantic_support,
                "coverage_axis_components": coverage_details.get("coverage_axis_components") or {},
                "coherence_proxy": audit.get("coherence_proxy"),
                "separation_margin": audit.get("separation_margin"),
                "style_purity_proxy": audit.get("style_purity_proxy"),
                "domain_purity_proxy": audit.get("domain_purity_proxy"),
            }
            certification_passes.append(ready)
            continue
        if metric_name == "small_lm_probe_gain_score":
            details = meta.get("utility_probe_details") or {}
            aggregate = details.get("aggregate") or {}
            protocol = details.get("protocol") or {}
            evidence_summary = aggregate.get("utility_evidence_summary") or {}
            certification_shadow = aggregate.get("certification_shadow") or {}
            pass_statistic = str(aggregate.get("pass_statistic") or protocol.get("utility_pass_statistic") or "")
            canonical_baseline = str(aggregate.get("canonical_baseline") or protocol.get("canonical_baseline") or "")
            canonical_in_domain = (details.get("in_domain") or {}).get(canonical_baseline) or {}
            min_gain = evidence_summary.get("strict_min_gain", aggregate.get("reported_small_lm_probe_gain_score_min"))
            min_delta = evidence_summary.get("strict_min_delta_nll", aggregate.get("min_delta_nll"))
            min_ci_low = evidence_summary.get("strict_min_delta_nll_ci_low", aggregate.get("min_delta_nll_ci_low"))
            if isinstance(evidence_summary.get("certification_ready"), bool):
                ready = bool(evidence_summary.get("certification_ready"))
            elif isinstance(certification_shadow.get("certification_ready"), bool):
                ready = bool(certification_shadow.get("certification_ready"))
            else:
                ready = bool(
                    stage_c.get("utility_axis_pass")
                    and canonical_baseline == "baseline_multi_matched_stageA_random"
                    and pass_statistic == "min"
                    and isinstance(min_gain, (int, float)) and float(min_gain) > 0.0
                    and isinstance(min_delta, (int, float)) and float(min_delta) > 0.0
                    and isinstance(min_ci_low, (int, float)) and float(min_ci_low) > 0.0
                )
            readiness_by_dataset[str(dataset)] = {
                "certification_ready": ready,
                "development_pass": bool(stage_c.get("utility_axis_pass")),
                "final_certification_scope": (
                    evidence_summary.get("final_certification_scope")
                    or aggregate.get("final_certification_scope")
                    or stage_c.get("final_certification_scope")
                ),
                "final_scope_certification_ready": evidence_summary.get(
                    "final_scope_certification_ready",
                    aggregate.get("final_scope_certification_ready"),
                ),
                "in_domain_certification_ready": evidence_summary.get(
                    "in_domain_certification_ready",
                    aggregate.get("in_domain_certification_ready"),
                ),
                "cross_domain_certification_ready": evidence_summary.get(
                    "cross_domain_certification_ready",
                    aggregate.get("cross_domain_certification_ready"),
                ),
                "domain_specific_certification_ready": evidence_summary.get(
                    "domain_specific_certification_ready",
                    aggregate.get("domain_specific_certification_ready"),
                ),
                "general_purpose_certification_ready": evidence_summary.get(
                    "general_purpose_certification_ready",
                    aggregate.get("general_purpose_certification_ready"),
                ),
                "in_domain_utility_axis_pass": evidence_summary.get(
                    "in_domain_utility_axis_pass",
                    stage_c.get("in_domain_utility_axis_pass"),
                ),
                "cross_domain_utility_axis_pass": evidence_summary.get(
                    "cross_domain_utility_axis_pass",
                    stage_c.get("cross_domain_utility_axis_pass"),
                ),
                "final_utility_axis_pass": evidence_summary.get(
                    "final_utility_axis_pass",
                    stage_c.get("final_utility_axis_pass"),
                ),
                "pass_statistic": pass_statistic,
                "canonical_baseline": canonical_baseline,
                "diagnostic_baselines": aggregate.get("diagnostic_baselines") or protocol.get("diagnostic_baselines"),
                "failure_analysis": aggregate.get("utility_failure_analysis") or {},
                "utility_evidence_summary": evidence_summary,
                "canonical_mean_gain": evidence_summary.get("canonical_mean_gain"),
                "signal_status": evidence_summary.get("signal_status"),
                "signal_status_reason": evidence_summary.get("signal_status_reason"),
                "in_domain_signal_status": evidence_summary.get("in_domain_signal_status"),
                "ood_signal_status": evidence_summary.get("ood_signal_status"),
                "signal_interpretation": evidence_summary.get("signal_interpretation") or {},
                "reported_small_lm_probe_gain_score_min": min_gain,
                "min_delta_nll": min_delta,
                "min_delta_nll_ci_low": min_ci_low,
                "max_minimum_detectable_delta_nll_95": evidence_summary.get(
                    "max_minimum_detectable_delta_nll_95",
                    aggregate.get("max_minimum_detectable_delta_nll_95"),
                ),
                "min_effect_to_mde_ratio": evidence_summary.get(
                    "min_effect_to_mde_ratio",
                    aggregate.get("min_effect_to_mde_ratio"),
                ),
                "min_detectable_effect_fraction": evidence_summary.get(
                    "min_detectable_effect_fraction",
                    aggregate.get("min_detectable_effect_fraction"),
                ),
                "strict_min_relative_nll_gain": evidence_summary.get("strict_min_relative_nll_gain"),
                "worst_in_domain_gain": evidence_summary.get("worst_in_domain_gain"),
                "worst_in_domain_delta_nll": evidence_summary.get("worst_in_domain_delta_nll"),
                "worst_ood_gain": evidence_summary.get("worst_ood_gain"),
                "worst_ood_delta_nll": evidence_summary.get("worst_ood_delta_nll"),
                "worst_ood_pair": evidence_summary.get("worst_ood_pair"),
                "protocol_blocker_count": evidence_summary.get("protocol_blocker_count"),
                "signal_blocker_count": evidence_summary.get("signal_blocker_count"),
                "stress_reported_small_lm_probe_gain_score_min": aggregate.get("stress_reported_small_lm_probe_gain_score_min"),
                "probe_model_name": protocol.get("probe_model_name"),
                "max_train_steps": protocol.get("max_train_steps"),
                "train_epochs": protocol.get("train_epochs"),
                "selected_effective_train_steps_mean": canonical_in_domain.get("selected_effective_train_steps_mean"),
                "baseline_effective_train_steps_mean": canonical_in_domain.get("baseline_effective_train_steps_mean"),
                "selected_train_token_exposure_ratio_mean": canonical_in_domain.get("selected_train_token_exposure_ratio_mean"),
                "baseline_train_token_exposure_ratio_mean": canonical_in_domain.get("baseline_train_token_exposure_ratio_mean"),
                "selected_target_train_exposure_ratio_mean": canonical_in_domain.get("selected_target_train_exposure_ratio_mean"),
                "baseline_target_train_exposure_ratio_mean": canonical_in_domain.get("baseline_target_train_exposure_ratio_mean"),
                "evidence_tier": evidence_summary.get("evidence_tier") or certification_shadow.get("evidence_tier"),
                "scope_snapshots": certification_shadow.get("scope_snapshots") or {},
                "worst_cells": certification_shadow.get("worst_cells") or {},
                "stability_analysis": certification_shadow.get("stability_analysis") or {},
                "step_cap_analysis": certification_shadow.get("step_cap_analysis") or {},
                "certification_shadow": certification_shadow,
                "certification_blockers": evidence_summary.get("certification_blockers") or certification_shadow.get("blockers") or [],
                "protocol_readiness": certification_shadow.get("protocol_readiness") or {},
                "in_domain_signal": certification_shadow.get("in_domain_signal") or {},
                "ood_signal": certification_shadow.get("ood_signal") or {},
                "blocker_categories": certification_shadow.get("blocker_categories") or {},
                "ood_pair_count": evidence_summary.get("ood_pair_count", aggregate.get("ood_pair_count")),
                "ood_expected_pair_count": evidence_summary.get("ood_expected_pair_count", aggregate.get("ood_expected_pair_count")),
                "ood_eval_datasets": aggregate.get("ood_eval_datasets") or [],
            }
            certification_passes.append(ready)
    if readiness_by_dataset:
        cert_label = "Pass" if all(certification_passes) else ("Fail" if not any(certification_passes) else "Mixed")
        final_scopes = sorted(
            {
                str(meta.get("final_certification_scope"))
                for meta in readiness_by_dataset.values()
                if meta.get("final_certification_scope")
            }
        )
        payload["certification_label"] = cert_label
        payload["certification_ready"] = bool(cert_label == "Pass")
        payload["final_certification_scopes"] = final_scopes
        payload["final_certification_scope"] = final_scopes[0] if len(final_scopes) == 1 else None
        payload["in_domain_certification_ready_count"] = sum(
            1 for meta in readiness_by_dataset.values() if bool(meta.get("in_domain_certification_ready"))
        )
        payload["cross_domain_certification_ready_count"] = sum(
            1 for meta in readiness_by_dataset.values() if bool(meta.get("cross_domain_certification_ready"))
        )
        payload["domain_specific_certification_ready_count"] = sum(
            1 for meta in readiness_by_dataset.values() if bool(meta.get("domain_specific_certification_ready"))
        )
        payload["general_purpose_certification_ready_count"] = sum(
            1 for meta in readiness_by_dataset.values() if bool(meta.get("general_purpose_certification_ready"))
        )
        payload["certification_datasets"] = readiness_by_dataset
    return payload


def _score_means(metric_name: str, scoring_manifest: Dict[str, Any], run_summary: Dict[str, Any]) -> Dict[str, float | None]:
    if metric_name in CORE_SUBSET_METRICS:
        subset_profile = _summary_profile_datasets(run_summary)
        return {
            str(dataset_name): meta.get(metric_name)
            for dataset_name, meta in subset_profile.items()
            if not str(dataset_name).startswith("_") and isinstance(meta, dict)
        }
    means: Dict[str, float | None] = {}
    for dataset_name, meta in (scoring_manifest.get("datasets") or {}).items():
        metric_group = meta.get("core_metrics") if metric_name in CORE_SELECTION_METRICS else meta.get("diagnostic_metrics")
        stat = (metric_group or {}).get(metric_name) or {}
        means[str(dataset_name)] = stat.get("mean")
    return means


def build_metric_maturity_snapshot(
    write_path: Path = METRIC_MATURITY_SNAPSHOT_PATH,
    *,
    scoring_manifest_path: Path = SCORING_MANIFEST_PATH,
    run_summary_path: Path = RUN_SUMMARY_PATH,
    validation_report_path: Path = VALIDATION_REPORT_PATH,
) -> Path:
    scoring_manifest = load_json(scoring_manifest_path)
    run_summary = load_json(run_summary_path)
    validation_report = load_json(validation_report_path)
    metric_spec = load_json(METRIC_SPEC_PATH)
    property_reports = _load_property_reports()
    fail_reason_counts = _profile_fail_reasons_by_metric(property_reports)

    rows: List[Dict[str, Any]] = []
    for metric_name in TRACKED_METRICS:
        policy = MATURITY_POLICY[metric_name]
        metric_meta = (metric_spec.get("metrics") or {}).get(metric_name) or {}
        metric_status = str(metric_meta.get("status") or "")
        assertion_details = _metric_assertion_details(metric_name, property_reports)
        if metric_name in CORE_SUBSET_METRICS:
            validation = _subset_metric_validation(validation_report, metric_name, run_summary)
        else:
            validation = _property_assertion_validation(metric_name, property_reports)
        means = _score_means(metric_name, scoring_manifest, run_summary)
        identity_label, identity_reason = _identity_gate(metric_name, assertion_details, validation=validation)
        stability_label, stability_reason = _stability_gate(metric_name, means)
        impact_label, impact_reason = _impact_gate(metric_name, property_reports, fail_reason_counts)
        gates = {
            "identity": identity_label,
            "stability": stability_label,
            "impact": impact_label,
        }
        if metric_name in {"subset_coverage_retention_score", "small_lm_probe_gain_score"}:
            gates["certification"] = str(validation.get("certification_label") or "Mixed")
        gate_reasons = {
            "identity": identity_reason,
            "stability": stability_reason,
            "impact": impact_reason,
        }
        if metric_name == "subset_coverage_retention_score":
            gate_reasons["certification"] = (
                "coverage certification requires Stage C coverage pass plus cluster_backbone_audit.passed on every dataset"
            )
        elif metric_name == "small_lm_probe_gain_score":
            gate_reasons["certification"] = (
                "utility certification requires protocol readiness plus in-domain and all-pairwise OOD strict signals"
            )
        completion_level = _completion_level(gates)
        if metric_status in {"diagnostic", "deprecated_diagnostic"} and completion_level == "High":
            completion_level = "Mid"
        completion_action = {
            "High": "Freeze",
            "Mid": "Validate",
            "Low": "Replace",
        }[completion_level]
        if metric_name in {"subset_coverage_retention_score", "small_lm_probe_gain_score"} and not bool(validation.get("certification_ready")):
            completion_action = "Certify"
        rows.append(
            {
                "metric": metric_name,
                "role": _role_for_metric(metric_name),
                "maturity": completion_level,
                "action": completion_action,
                "current_method": policy["method"],
                "dataset_means": means,
                "validation": validation["label"],
                "last_updated": datetime.now(timezone.utc).date().isoformat(),
                "notes": policy["notes"],
                "evidence": {
                    "validation": validation,
                    "assertions": assertion_details,
                    "benchmark_audits": _benchmark_audit_evidence(metric_name, property_reports),
                    "gates": gates,
                    "gate_reasons": gate_reasons,
                    "profile_fail_reason_count": int(fail_reason_counts.get(metric_name) or 0),
                    "metric_status": metric_status,
                },
            }
        )

    payload = {
        "schema_version": "metric-maturity-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_spec_path": str(METRIC_SPEC_PATH),
        "metric_spec_fingerprint": fingerprint_files([METRIC_SPEC_PATH]),
        "validation_summary": validation_report.get("summary"),
        "tracked_metrics": rows,
        "tracker_design": {
            "canonical_core": list(CORE_SELECTION_METRICS),
            "subset_core": list(CORE_SUBSET_METRICS),
            "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
            "subset_evaluator": "subset_coverage_retention_score",
        },
        "metric_spec_status_map": {
            metric: (metric_spec.get("metrics") or {}).get(metric, {}).get("status")
            for metric in metric_spec.get("metrics", {})
        },
    }
    save_json(write_path, payload)
    return write_path
