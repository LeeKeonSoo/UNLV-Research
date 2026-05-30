#!/usr/bin/env python3
"""Validation helpers for the generic data evaluation pipeline."""

from __future__ import annotations

import json
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
LEGACY_VARIANT_PROFILE_ORDER = ("strict", "balanced", "coverage_preserving")
METRIC_ROLES = {"gate", "selection_signal", "subset_validator", "diagnostic"}
METRIC_STATUSES = {"paper_backed", "paper_aligned", "diagnostic", "deprecated_diagnostic"}
ORTHOGONALITY_MAX_ABS_SPEARMAN = 0.92
ORTHOGONALITY_SAMPLE_LIMIT = 30000
THEORY_AXIS_EXPECTED = {
    "structural_validity_gate": "Validity",
    "structural_validity_score": "Validity",
    "reference_quality_score": "Quality",
    "exact_duplicate_indicator": "Redundancy",
    "shingle_near_duplicate_indicator": "Redundancy",
    "shingle_near_duplicate_risk_score": "Redundancy",
    "subset_coverage_retention_score": "Coverage",
    "small_lm_probe_gain_score": "Utility",
}


@dataclass
class ValidationItem:
    name: str
    ok: bool
    details: Dict[str, Any]


def _count_lines(path: Path) -> int:
    return count_nonempty_lines_resilient(path)


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


def validate_outputs() -> List[ValidationItem]:
    items: List[ValidationItem] = []
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

    run_manifest = load_json(RUN_MANIFEST_PATH)
    run_summary = load_json(RUN_SUMMARY_PATH)
    scoring_manifest = load_json(SCORING_MANIFEST_PATH)
    utility_probe_results = load_json(UTILITY_PROBE_RESULTS_PATH)
    utility_sensitivity_audit = load_json(UTILITY_SENSITIVITY_AUDIT_PATH) if UTILITY_SENSITIVITY_AUDIT_PATH.exists() else {}
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
                    and isinstance(root.get("selector_tuning_allowed"), bool),
                    details={"dataset": dataset_name, "probe_sensitivity": sensitivity, "root_cause_decision": root},
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
                    "baseline_multi_matched_stageA_random",
                    "baseline_style_matched_stageA_random",
                    "baseline_length_matched_stageA_random",
                    "baseline_quality_band_matched_stageA_random",
                }
                diagnostic_matched_baselines = expected_matched_baselines - {"baseline_multi_matched_stageA_random"}
                in_domain = utility_details.get("in_domain") or {}
                failure_analysis = utility_aggregate.get("utility_failure_analysis") or {}
                items.append(
                    ValidationItem(
                        name=f"utility_matched_diagnostic_baselines_present_{profile_name}_{dataset}",
                        ok=diagnostic_matched_baselines.issubset(diagnostic_baselines)
                        and all(isinstance(in_domain.get(name), dict) for name in expected_matched_baselines)
                        and isinstance(failure_analysis.get("matched_baseline_deltas"), dict)
                        and isinstance(failure_analysis.get("failure_mode"), str),
                        details={
                            "diagnostic_baselines": sorted(diagnostic_baselines),
                            "in_domain_keys": sorted(str(name) for name in in_domain.keys()),
                            "failure_analysis": failure_analysis,
                        },
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
                        and baseline_control_policy.get("canonical_matching_policy") == "quality_length_style_domain_with_hierarchical_fallback"
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
                        and all(isinstance(evidence_summary.get(name), bool) for name in evidence_required_bool_fields)
                        and all(isinstance(evidence_summary.get(name), int) for name in evidence_required_int_fields)
                        and evidence_summary.get("evidence_tier")
                        in {
                            "development_only",
                            "in_domain_strict_signal",
                            "cross_domain_strict_signal",
                            "certification_ready",
                            "invalid_probe_evidence",
                            "random_baseline_gain",
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
                        and isinstance(evidence_summary.get("worst_ood_pair"), str)
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
                                probe_status.get("status") in {"valid", "invalid", "not_evaluated"}
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
                                in {"pass", "probe_invalid", "random_gain_only", "matched_inconclusive", "strict_negative"}
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
                    and isinstance(in_domain.get("baseline_multi_matched_stageA_random"), dict)
                )
                has_ood_baselines = (
                    isinstance(out_of_domain, dict)
                    and actual_ood_eval_datasets == expected_ood_eval_datasets
                    and all(
                        isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_full_random"), dict)
                        and isinstance((out_of_domain.get(eval_dataset) or {}).get("baseline_stageA_random"), dict)
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
                    ok=isinstance(cluster_backbone_audit.get("passed"), bool),
                    details={"cluster_backbone_audit": cluster_backbone_audit},
                )
            )

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


def main(write_report: Path | None = VALIDATION_REPORT_PATH) -> int:
    items = validate_outputs()
    passed = [x for x in items if x.ok]
    failed = [x for x in items if not x.ok]
    report = {
        "schema_version": SCHEMA_VERSION,
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
