#!/usr/bin/env python3
"""Build a Core/Policy proxy alignment diagnostic report.

This report is diagnostic-only. It uses Stage-C evidence to explain where Core
feature-space gains do or do not align with small-LM Utility, but it never turns
Utility into a Stage-B selector objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


SELECTOR_BASELINE_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
UTILITY_POWER_SWEEP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_probe_power_sweep.json"
DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "core_proxy_alignment_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_proxy_alignment_report.md"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _selector_verdict(selector_audit: Dict[str, Any], dataset: str, baseline: str) -> Dict[str, Any]:
    comparison = (
        ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
        .get("comparisons", {})
        .get(str(baseline), {})
    )
    verdict = comparison.get("verdict") if isinstance(comparison, dict) else None
    return verdict if isinstance(verdict, dict) else {}


def _numeric_gap(selector_audit: Dict[str, Any], dataset: str, baseline: str, metric: str) -> Dict[str, Any]:
    comparison = (
        ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
        .get("comparisons", {})
        .get(str(baseline), {})
    )
    numeric = comparison.get("numeric_comparison") if isinstance(comparison, dict) else None
    payload = (numeric or {}).get(str(metric)) if isinstance(numeric, dict) else None
    return payload if isinstance(payload, dict) else {}


def _power_summary(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    decision = (((power_sweep.get("datasets") or {}).get(str(dataset)) or {}).get("decision") or {})
    return {
        "available": bool(decision),
        "compatible_runs": int(decision.get("compatible_runs") or 0),
        "probe_valid_runs": int(decision.get("probe_valid_runs") or 0),
        "selected_gt_random_runs": int(decision.get("selected_gt_random_runs") or 0),
        "stable_probe_valid": bool(decision.get("stable_probe_valid")),
    }


def _easy_nll_tension(memorization_gap: Dict[str, Any]) -> Dict[str, Any]:
    repeat_delta = _safe_float(memorization_gap.get("repeat_pressure_delta"))
    word_delta = _safe_float(memorization_gap.get("word_count_delta"))
    useful_recurrence_delta = _safe_float(memorization_gap.get("useful_recurrence_delta"))
    quality_delta = _safe_float(memorization_gap.get("quality_delta"))
    redundancy_delta = _safe_float(memorization_gap.get("redundancy_risk_delta"))
    baseline_more_repetitive = bool(repeat_delta <= -0.05)
    baseline_longer = bool(word_delta <= -10.0)
    selected_core_better = bool(quality_delta > 0.0 and redundancy_delta < 0.0)
    return {
        "candidate": bool(selected_core_better and (baseline_more_repetitive or baseline_longer)),
        "baseline_more_repetitive": baseline_more_repetitive,
        "baseline_longer": baseline_longer,
        "selected_core_better": selected_core_better,
        "repeat_pressure_delta": round(repeat_delta, 6),
        "word_count_delta": round(word_delta, 6),
        "useful_recurrence_delta": round(useful_recurrence_delta, 6),
        "quality_delta": round(quality_delta, 6),
        "redundancy_risk_delta": round(redundancy_delta, 6),
    }


def _alignment_status(
    *,
    framework_status: str,
    transfer_category: str,
    easy_nll: Dict[str, Any],
    power: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        framework_status in {"stage_c_development_ready", "stage_c_development_ready_with_token_exposure_caveat"}
        or transfer_category in {"stage_c_development_ready", "stage_c_development_ready_with_token_exposure_caveat"}
    ):
        return {
            "status": (
                "stage_c_development_ready_with_token_exposure_caveat"
                if framework_status == "stage_c_development_ready_with_token_exposure_caveat"
                or transfer_category == "stage_c_development_ready_with_token_exposure_caveat"
                else "stage_c_development_ready"
            ),
            "selector_policy_action": "hold",
            "next_step": "Run certification-grade Stage-C Utility follow-up before promoting this profile globally.",
        }
    if framework_status == "utility_probe_preset_instability" or transfer_category == "probe_preset_candidate_available":
        return {
            "status": "probe_preset_instability_with_candidate",
            "selector_policy_action": "hold",
            "next_step": "Standardize the valid stronger Utility preset and rerun Stage-C before Core proxy tuning.",
        }
    if framework_status == "utility_probe_not_interpretable":
        return {
            "status": "not_diagnosable_until_probe_valid",
            "selector_policy_action": "hold",
            "next_step": "Redesign or strengthen Utility positive/random/destructive controls before Core proxy tuning.",
        }
    if framework_status == "core_policy_proxy_not_utility_supported":
        return {
            "status": (
                "core_proxy_utility_mismatch_with_easy_nll_tension"
                if easy_nll.get("candidate")
                else "core_proxy_utility_mismatch"
            ),
            "selector_policy_action": "inspect_core_proxy_calibration",
            "next_step": (
                "Inspect whether Quality/Redundancy/Learnability proxies remove repetition or length signal that the "
                "small-LM probe rewards; do not add Utility to Stage-B objective."
            ),
        }
    if framework_status == "strict_baseline_confounded_by_easy_nll_signal":
        return {
            "status": "strict_baseline_easy_nll_confound_supported",
            "selector_policy_action": "hold",
            "next_step": "Promote repeat-pressure matched control to a reported strict-baseline diagnostic before selector changes.",
        }
    if transfer_category == "lm_train_memorization_proxy_gap" or easy_nll.get("candidate"):
        return {
            "status": "easy_nll_confound_candidate",
            "selector_policy_action": "hold",
            "next_step": "Run targeted anti-memorization Utility diagnostic for this dataset.",
        }
    if power.get("stable_probe_valid") and power.get("selected_gt_random_runs", 0) > 0:
        return {
            "status": "core_proxy_partially_supported_by_utility",
            "selector_policy_action": "hold",
            "next_step": "Keep Core-Metric-Policy fixed and focus on strict counterfactual certification.",
        }
    return {
        "status": "alignment_unresolved",
        "selector_policy_action": "diagnose",
        "next_step": "Inspect selector feature audit and Utility transfer-gap report together.",
    }


def _dataset_report(
    *,
    dataset: str,
    selector_audit: Dict[str, Any],
    transfer_report: Dict[str, Any],
    power_sweep: Dict[str, Any],
) -> Dict[str, Any]:
    transfer_payload = ((transfer_report.get("datasets") or {}).get(str(dataset)) or {})
    transfer_gap = transfer_payload.get("transfer_gap") or {}
    implication = transfer_gap.get("framework_implication") or {}
    memorization_gap = transfer_gap.get("matched_memorization_proxy_gap") or {}
    easy_nll = _easy_nll_tension(memorization_gap)
    power = _power_summary(power_sweep, dataset)
    status = _alignment_status(
        framework_status=str(implication.get("status") or ""),
        transfer_category=str(transfer_gap.get("category") or ""),
        easy_nll=easy_nll,
        power=power,
    )
    metrics = {
        "stageA_random": _selector_verdict(selector_audit, dataset, "stageA_random"),
        "multi_matched_stageA_random": _selector_verdict(selector_audit, dataset, "multi_matched_stageA_random"),
    }
    matched_numeric = {
        name: _numeric_gap(selector_audit, dataset, "multi_matched_stageA_random", name)
        for name in (
            "quality_score",
            "learnability_core",
            "redundancy_risk",
            "word_count",
            "intra_chunk_repeat_pressure",
            "useful_recurrence_score",
            "lexical_diversity",
            "validity_warning_count",
        )
    }
    return {
        "dataset": str(dataset),
        "selector_objective_scope": "Core metrics only; Utility remains Stage-C diagnostic/validation evidence",
        "transfer_category": transfer_gap.get("category"),
        "framework_implication": implication,
        "alignment": status,
        "easy_nll_tension": easy_nll,
        "selector_feature_verdicts": metrics,
        "multi_matched_numeric_gaps": matched_numeric,
        "power_sweep": power,
    }


def build_report(
    *,
    selector_audit: Dict[str, Any],
    transfer_report: Dict[str, Any],
    power_sweep: Dict[str, Any],
) -> Dict[str, Any]:
    datasets = sorted(
        str(dataset)
        for dataset in (transfer_report.get("datasets") or {}).keys()
    )
    payloads = {
        dataset: _dataset_report(
            dataset=dataset,
            selector_audit=selector_audit,
            transfer_report=transfer_report,
            power_sweep=power_sweep,
        )
        for dataset in datasets
    }
    status_counts: Dict[str, int] = {}
    for payload in payloads.values():
        status = str(((payload.get("alignment") or {}).get("status")) or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": "core-proxy-alignment-report-v1",
        "purpose": (
            "Diagnose whether Stage-B Core feature-space gains align with Stage-C Utility evidence, "
            "without using Utility as a selector objective."
        ),
        "inputs": {
            "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
            "utility_transfer_gap_report": str(UTILITY_TRANSFER_GAP_REPORT_PATH),
            "utility_probe_power_sweep": str(UTILITY_POWER_SWEEP_REPORT_PATH),
        },
        "summary": {
            "dataset_count": int(len(payloads)),
            "alignment_status_counts": status_counts,
        },
        "datasets": payloads,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Core Proxy Alignment Report",
        "",
        "This diagnostic checks whether Core feature gains conflict with easy-NLL signals such as length or repetition pressure.",
        "",
        "| Dataset | Transfer category | Alignment | Easy-NLL tension | Repeat gap | Length gap | Selector action | Next step |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        alignment = payload.get("alignment") or {}
        easy = payload.get("easy_nll_tension") or {}
        lines.append(
            f"| {dataset} | {payload.get('transfer_category')} | {alignment.get('status')} | "
            f"{easy.get('candidate')} | {float(easy.get('repeat_pressure_delta') or 0):+.6f} | "
            f"{float(easy.get('word_count_delta') or 0):+.3f} | "
            f"{alignment.get('selector_policy_action')} | {alignment.get('next_step')} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Core proxy alignment diagnostic report.")
    parser.add_argument("--selector-audit", type=Path, default=SELECTOR_BASELINE_AUDIT_PATH)
    parser.add_argument("--transfer-report", type=Path, default=UTILITY_TRANSFER_GAP_REPORT_PATH)
    parser.add_argument("--power-sweep", type=Path, default=UTILITY_POWER_SWEEP_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        selector_audit=load_json(args.selector_audit),
        transfer_report=load_json(args.transfer_report),
        power_sweep=load_json(args.power_sweep) if args.power_sweep.exists() else {},
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[23] core proxy alignment json: {args.output}", flush=True)
    print(f"[23] core proxy alignment md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
