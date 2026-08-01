#!/usr/bin/env python3
"""Build a diagnostic report for Core proxy calibration targets.

This report converts Stage-C mismatch evidence into Core/Policy audit targets.
It is diagnostic-only: Utility remains Stage-C validation evidence and is not
added to the Stage-B selector objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


SELECTOR_BASELINE_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
CORE_PROXY_ALIGNMENT_REPORT_PATH = OUTPUT_DIR / "validation" / "core_proxy_alignment_report.json"
POLICY_ABLATION_AUDIT_PATH = OUTPUT_DIR / "validation" / "policy_ablation_audit.json"
DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "core_proxy_calibration_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_proxy_calibration_report.md"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _numeric_gap(selector_audit: Dict[str, Any], dataset: str, metric: str) -> Dict[str, Any]:
    payload = (
        ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
        .get("comparisons", {})
        .get("multi_matched_stageA_random", {})
    )
    numeric = payload.get("numeric_comparison") if isinstance(payload, dict) else {}
    value = (numeric or {}).get(str(metric))
    return value if isinstance(value, dict) else {}


def _gap_delta(selector_audit: Dict[str, Any], dataset: str, metric: str) -> float:
    return _safe_float(_numeric_gap(selector_audit, dataset, metric).get("delta_selected_minus_baseline"))


def _gap_std(selector_audit: Dict[str, Any], dataset: str, metric: str) -> float:
    return _safe_float(_numeric_gap(selector_audit, dataset, metric).get("standardized_delta"))


def _ablation_summary(policy_ablation: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    variants = (((policy_ablation.get("datasets") or {}).get(str(dataset)) or {}).get("variant_results") or {})
    if not isinstance(variants, dict):
        variants = {}
    canonical = variants.get("canonical") or {}
    closest = []
    for name, payload in variants.items():
        if str(name) == "canonical" or not isinstance(payload, dict):
            continue
        delta = payload.get("delta_vs_canonical") or {}
        closest.append(
            {
                "variant": str(name),
                "jaccard_vs_canonical": _safe_float((payload.get("high_quality_recovery_vs_canonical") or {}).get("jaccard_vs_canonical")),
                "delta_coverage": _safe_float(delta.get("coverage_score")),
                "delta_quality": _safe_float(delta.get("mean_quality")),
                "delta_learnability": _safe_float(delta.get("mean_learnability_support")),
                "delta_redundancy_risk": _safe_float(delta.get("mean_redundancy_risk")),
                "delta_predictive_utility_proxy": _safe_float(delta.get("mean_predictive_utility_proxy")),
                "delta_length_similarity": _safe_float(delta.get("length_distribution_similarity")),
            }
        )
    closest.sort(key=lambda item: (-abs(item["delta_predictive_utility_proxy"]), item["variant"]))
    return {
        "available": bool(variants),
        "canonical": {
            "coverage_score": canonical.get("coverage_score"),
            "mean_quality": canonical.get("mean_quality"),
            "mean_learnability_support": canonical.get("mean_learnability_support"),
            "mean_redundancy_risk": canonical.get("mean_redundancy_risk"),
            "mean_predictive_utility_proxy": canonical.get("mean_predictive_utility_proxy"),
            "length_distribution_similarity": canonical.get("length_distribution_similarity"),
            "learnability_swap_count": canonical.get("learnability_swap_count"),
            "accepted_by_counts": canonical.get("accepted_by_counts"),
        },
        "variant_deltas": closest,
    }


def _candidate_variant_recommendation(policy_ablation: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    variants = (((policy_ablation.get("datasets") or {}).get(str(dataset)) or {}).get("variant_results") or {})
    if not isinstance(variants, dict) or "canonical" not in variants:
        return {
            "available": False,
            "recommended_variant": None,
            "reason": "Policy ablation audit is unavailable for this dataset.",
        }
    candidates = []
    for name, payload in variants.items():
        if str(name) == "canonical" or not isinstance(payload, dict):
            continue
        delta = payload.get("delta_vs_canonical") or {}
        coverage_delta = _safe_float(delta.get("coverage_score"))
        quality_delta = _safe_float(delta.get("mean_quality"))
        learnability_delta = _safe_float(delta.get("mean_learnability_support"))
        predictive_delta = _safe_float(delta.get("mean_predictive_utility_proxy"))
        length_similarity_delta = _safe_float(delta.get("length_distribution_similarity"))
        jaccard = _safe_float((payload.get("high_quality_recovery_vs_canonical") or {}).get("jaccard_vs_canonical"))
        if coverage_delta < -0.01:
            continue
        score = (
            (quality_delta * 4.0)
            + (max(0.0, -learnability_delta) * 2.0)
            + (max(0.0, -predictive_delta) * 1.5)
            + (length_similarity_delta * 0.5)
            - (max(0.0, -coverage_delta) * 4.0)
        )
        candidates.append(
            {
                "variant": str(name),
                "score": round(float(score), 6),
                "coverage_delta": round(coverage_delta, 6),
                "quality_delta": round(quality_delta, 6),
                "learnability_delta": round(learnability_delta, 6),
                "predictive_utility_proxy_delta": round(predictive_delta, 6),
                "length_similarity_delta": round(length_similarity_delta, 6),
                "jaccard_vs_canonical": round(jaccard, 6),
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["variant"])))
    if not candidates:
        return {
            "available": True,
            "recommended_variant": None,
            "reason": "No ablation variant preserved coverage while targeting the mismatch axes.",
            "candidates": [],
        }
    return {
        "available": True,
        "recommended_variant": candidates[0]["variant"],
        "reason": (
            "Recommended among Core-only ablations that preserve coverage while reducing reliance on "
            "learnability/predictive proxy axes implicated by Stage-C mismatch."
        ),
        "candidates": candidates,
    }


def _calibration_targets(
    *,
    selector_audit: Dict[str, Any],
    transfer_gap: Dict[str, Any],
    alignment: Dict[str, Any],
    dataset: str,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    anti = transfer_gap.get("anti_memorization_diagnostic_baseline") or {}
    power = transfer_gap.get("power_sweep") or {}
    align_status = str((alignment.get("alignment") or {}).get("status") or "")
    quality_delta = _gap_delta(selector_audit, dataset, "quality")
    learnability_delta = _gap_delta(selector_audit, dataset, "learnability_support")
    predictive_delta = _gap_delta(selector_audit, dataset, "diagnostic_predictive_utility")
    repeat_delta = _gap_delta(selector_audit, dataset, "intra_chunk_repeat_pressure")
    recurrence_delta = _gap_delta(selector_audit, dataset, "useful_recurrence_score")
    length_delta = _gap_delta(selector_audit, dataset, "word_count")
    redundancy_delta = _gap_delta(selector_audit, dataset, "redundancy_risk")

    if align_status in {"stage_c_development_ready", "stage_c_development_ready_with_token_exposure_caveat"}:
        return [
            {
                "target": "stage_c_certification_followup",
                "priority": 1,
                "reason": (
                    "Current Stage-C development validation passes, so the next step is certification-grade "
                    "Utility follow-up rather than another Core proxy change."
                ),
                "evidence": {
                    "transfer_category": transfer_gap.get("category"),
                    "stage_c_pass": transfer_gap.get("stage_c_pass"),
                },
                "next_experiment": "Rerun Stage-C with certification-grade Utility budget and token-exposure diagnostics before global profile promotion.",
            }
        ]
    if align_status == "probe_preset_instability_with_candidate":
        return [
            {
                "target": "probe_preset_standardization",
                "priority": 1,
                "reason": (
                    "Power sweep found valid selected>Stage-A-random Utility presets, but the default "
                    "probe evidence remains unstable."
                ),
                "evidence": {
                    "transfer_category": transfer_gap.get("category"),
                    "best_valid_selected_gt_random_preset": (power or {}).get("best_valid_selected_gt_random_preset"),
                    "valid_selected_gt_random_presets": (power or {}).get("valid_selected_gt_random_presets"),
                },
                "next_experiment": "Standardize the stronger Utility preset and rerun Stage-C before changing Core policy.",
            }
        ]
    if align_status == "not_diagnosable_until_probe_valid":
        return [
            {
                "target": "probe_before_core_calibration",
                "priority": 1,
                "reason": "Utility probe evidence is not interpretable, so Core proxy calibration would be premature.",
                "evidence": {
                    "transfer_category": transfer_gap.get("category"),
                    "probe_valid": transfer_gap.get("probe_valid"),
                },
                "next_experiment": "Redesign or strengthen Utility sensitivity controls before changing Core proxy policy.",
            }
        ]
    if align_status == "strict_baseline_easy_nll_confound_supported":
        return [
            {
                "target": "strict_baseline_control_before_core_calibration",
                "priority": 1,
                "reason": (
                    "Repeat-pressure matched diagnostic supports selected chunks, so the next action is strict-baseline "
                    "control revision rather than Core proxy tuning."
                ),
                "evidence": {
                    "anti_mem_delta_nll": anti.get("delta_nll"),
                    "anti_mem_supports_selected": anti.get("supports_selected"),
                    "repeat_pressure_delta": round(repeat_delta, 6),
                    "word_count_delta": round(length_delta, 6),
                },
                "next_experiment": "Promote repeat-pressure/length matched strict controls as reported diagnostics before changing Stage-B policy.",
            }
        ]

    if (
        align_status == "core_proxy_utility_mismatch_with_easy_nll_tension"
        and bool(anti.get("available"))
        and not bool(anti.get("supports_selected"))
    ):
        targets.append(
            {
                "target": "learnability_proxy_semantics",
                "priority": 1,
                "reason": (
                    "Selected chunks have higher learnability/support proxies, but selected loses "
                    "the repeat-pressure matched anti-memorization diagnostic."
                ),
                "evidence": {
                    "learnability_delta": round(learnability_delta, 6),
                    "quality_delta": round(quality_delta, 6),
                    "predictive_utility_proxy_delta": round(predictive_delta, 6),
                    "anti_mem_delta_nll": anti.get("delta_nll"),
                    "valid_selected_gt_random_runs": power.get("valid_selected_gt_random_runs"),
                },
                "next_experiment": "Compare rejected/high-learnability candidates against selected chunks by template density, exercise structure, and held-out train/eval alignment.",
            }
        )
    if repeat_delta < -0.05 and recurrence_delta > 0.05 and redundancy_delta < 0.0:
        targets.append(
            {
                "target": "redundancy_useful_recurrence_calibration",
                "priority": 1,
                "reason": (
                    "Selected chunks reduce repeat pressure and redundancy risk while increasing useful recurrence, "
                    "but Utility does not support the selected subset."
                ),
                "evidence": {
                    "repeat_pressure_delta": round(repeat_delta, 6),
                    "useful_recurrence_delta": round(recurrence_delta, 6),
                    "redundancy_risk_delta": round(redundancy_delta, 6),
                    "repeat_pressure_standardized_delta": round(_gap_std(selector_audit, dataset, "intra_chunk_repeat_pressure"), 6),
                },
                "next_experiment": "Audit whether useful recurrence relief rewards surface structure that does not transfer, without rewarding memorization-heavy repetition.",
            }
        )
    if length_delta < -10.0:
        targets.append(
            {
                "target": "length_bucket_and_useful_length_support",
                "priority": 2,
                "reason": (
                    "Selected chunks remain materially shorter than the multi-matched baseline, "
                    "so length matching may be too coarse for this corpus."
                ),
                "evidence": {
                    "word_count_delta": round(length_delta, 6),
                    "word_count_standardized_delta": round(_gap_std(selector_audit, dataset, "word_count"), 6),
                },
                "next_experiment": "Test finer length buckets or a minimum useful-length floor in a diagnostic candidate profile.",
            }
        )
    if not targets:
        targets.append(
            {
                "target": "no_immediate_core_proxy_change",
                "priority": 3,
                "reason": "Current evidence does not isolate a Core proxy calibration target.",
                "evidence": {},
                "next_experiment": "Collect stronger Utility/probe evidence before changing Core policy.",
            }
        )
    targets.sort(key=lambda item: (int(item.get("priority") or 99), str(item.get("target") or "")))
    return targets


def _dataset_report(
    *,
    dataset: str,
    selector_audit: Dict[str, Any],
    transfer_report: Dict[str, Any],
    alignment_report: Dict[str, Any],
    policy_ablation: Dict[str, Any],
) -> Dict[str, Any]:
    transfer_gap = (((transfer_report.get("datasets") or {}).get(str(dataset)) or {}).get("transfer_gap") or {})
    alignment = ((alignment_report.get("datasets") or {}).get(str(dataset)) or {})
    targets = _calibration_targets(
        selector_audit=selector_audit,
        transfer_gap=transfer_gap,
        alignment=alignment,
        dataset=dataset,
    )
    metric_gaps = {
        metric: _numeric_gap(selector_audit, dataset, metric)
        for metric in (
            "quality",
            "learnability_support",
            "diagnostic_predictive_utility",
            "redundancy_risk",
            "intra_chunk_repeat_pressure",
            "useful_recurrence_score",
            "word_count",
            "lexical_diversity",
            "validity_warning_count",
        )
    }
    return {
        "dataset": str(dataset),
        "selector_objective_scope": "Core metrics only; Utility remains Stage-C validation/diagnostic evidence",
        "transfer_category": transfer_gap.get("category"),
        "alignment_status": (alignment.get("alignment") or {}).get("status"),
        "anti_memorization_supports_selected": (transfer_gap.get("anti_memorization_diagnostic_baseline") or {}).get("supports_selected"),
        "calibration_targets": targets,
        "multi_matched_metric_gaps": metric_gaps,
        "policy_ablation": _ablation_summary(policy_ablation, dataset),
        "candidate_variant_recommendation": _candidate_variant_recommendation(policy_ablation, dataset),
    }


def build_report(
    *,
    selector_audit: Dict[str, Any],
    transfer_report: Dict[str, Any],
    alignment_report: Dict[str, Any],
    policy_ablation: Dict[str, Any],
) -> Dict[str, Any]:
    datasets = sorted((transfer_report.get("datasets") or {}).keys())
    reports = {
        dataset: _dataset_report(
            dataset=dataset,
            selector_audit=selector_audit,
            transfer_report=transfer_report,
            alignment_report=alignment_report,
            policy_ablation=policy_ablation,
        )
        for dataset in datasets
    }
    target_counts: Dict[str, int] = {}
    for payload in reports.values():
        for target in payload.get("calibration_targets") or []:
            name = str(target.get("target") or "unknown")
            target_counts[name] = target_counts.get(name, 0) + 1
    return {
        "schema_version": "core-proxy-calibration-report-v1",
        "purpose": "Identify Core proxy calibration targets after Stage-C Utility mismatch, without changing selector objectives.",
        "inputs": {
            "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
            "utility_transfer_gap_report": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
            "core_proxy_alignment_report": str(CORE_PROXY_ALIGNMENT_REPORT_PATH),
            "policy_ablation_audit": str(POLICY_ABLATION_AUDIT_PATH),
        },
        "summary": {
            "dataset_count": int(len(reports)),
            "calibration_target_counts": target_counts,
        },
        "datasets": reports,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Core Proxy Calibration Report",
        "",
        "Diagnostic-only report. Utility is not used as a Stage-B selector objective.",
        "",
        "| Dataset | Transfer | Alignment | Anti-mem supports | Recommended variant | Priority targets |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        targets = payload.get("calibration_targets") or []
        recommendation = payload.get("candidate_variant_recommendation") or {}
        target_text = "; ".join(f"P{t.get('priority')} {t.get('target')}" for t in targets)
        lines.append(
            f"| {dataset} | {payload.get('transfer_category')} | {payload.get('alignment_status')} | "
            f"{payload.get('anti_memorization_supports_selected')} | {recommendation.get('recommended_variant')} | {target_text} |"
        )
    lines.append("")
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.extend([f"## {dataset}", ""])
        recommendation = payload.get("candidate_variant_recommendation") or {}
        lines.extend(
            [
                f"- Recommended Core-only variant: `{recommendation.get('recommended_variant')}`",
                f"- Recommendation reason: {recommendation.get('reason')}",
                "",
            ]
        )
        for target in payload.get("calibration_targets") or []:
            lines.extend(
                [
                    f"- Target: `{target.get('target')}`",
                    f"- Priority: `{target.get('priority')}`",
                    f"- Reason: {target.get('reason')}",
                    f"- Next experiment: {target.get('next_experiment')}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Core proxy calibration diagnostic report.")
    parser.add_argument("--selector-audit", type=Path, default=SELECTOR_BASELINE_AUDIT_PATH)
    parser.add_argument("--transfer-report", type=Path, default=UTILITY_TRANSFER_GAP_REPORT_PATH)
    parser.add_argument("--alignment-report", type=Path, default=CORE_PROXY_ALIGNMENT_REPORT_PATH)
    parser.add_argument("--policy-ablation", type=Path, default=POLICY_ABLATION_AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        selector_audit=load_json(args.selector_audit),
        transfer_report=load_json(args.transfer_report),
        alignment_report=load_json(args.alignment_report),
        policy_ablation=load_json(args.policy_ablation) if args.policy_ablation.exists() else {},
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[24] core proxy calibration json: {args.output}", flush=True)
    print(f"[24] core proxy calibration md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
