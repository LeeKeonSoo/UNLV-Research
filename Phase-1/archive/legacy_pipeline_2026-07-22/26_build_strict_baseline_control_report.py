#!/usr/bin/env python3
"""Build a Stage-C strict-baseline control decision report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "strict_baseline_control_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "strict_baseline_control_report.md"
CURATION_READINESS_REPORT_PATH = OUTPUT_DIR / "validation" / "curation_readiness_report.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
STAGE_C_PROTOCOL_DECISION_REPORT_PATH = OUTPUT_DIR / "validation" / "stage_c_protocol_decision_report.json"


def _load_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _anti_memorization_evidence(transfer_gap: Dict[str, Any], protocol: Dict[str, Any]) -> Dict[str, Any]:
    evidence = transfer_gap.get("anti_memorization_diagnostic_baseline")
    if not isinstance(evidence, dict):
        evidence = {}
    return {
        "available": bool(protocol.get("anti_memorization_diagnostic_available") or evidence.get("available")),
        "baseline": evidence.get("baseline"),
        "supports_selected": bool(protocol.get("anti_memorization_supports_selected") or evidence.get("supports_selected")),
        "delta_nll": evidence.get("delta_nll", protocol.get("anti_memorization_delta_nll")),
        "delta_nll_ci_low": evidence.get("delta_nll_ci_low"),
        "minimum_detectable_delta_nll_95_max": evidence.get("minimum_detectable_delta_nll_95_max"),
        "effect_to_mde_ratio_min": evidence.get("effect_to_mde_ratio_min"),
        "detectable_effect_fraction": evidence.get("detectable_effect_fraction"),
        "small_lm_probe_gain_score": evidence.get("small_lm_probe_gain_score"),
        "causal_mode": evidence.get("causal_mode"),
        "train_audit_gap": evidence.get("train_audit_gap"),
        "scope": "Stage C diagnostic only; never selector objective",
    }


def _strict_control_decision(dataset: str, readiness: Dict[str, Any], transfer: Dict[str, Any], protocol: Dict[str, Any]) -> Dict[str, Any]:
    stage_c = readiness.get("stage_c") or {}
    utility = readiness.get("utility") or {}
    implication = readiness.get("framework_implication") or {}
    transfer_gap = transfer.get("transfer_gap") or {}
    anti_evidence = _anti_memorization_evidence(transfer_gap, protocol)
    protocol_status = str(protocol.get("protocol_status") or "")
    anti_support = bool(protocol.get("anti_memorization_supports_selected"))
    token_caveat = bool(protocol.get("token_exposure_caveat"))
    canonical_selected_beats = bool(utility.get("selected_beats_multi_matched"))
    selected_beats_random = bool(utility.get("selected_beats_stageA_random") or protocol.get("utility_selected_beats_stageA_random"))
    coverage_passed = bool(stage_c.get("coverage_pass") or protocol.get("coverage_passed"))
    operational_total_effect_pass = bool(coverage_passed and selected_beats_random)
    stage_c_passed = bool(stage_c.get("passed"))
    replicated_families = protocol.get("replicated_valid_power_sweep_families") or []
    if not isinstance(replicated_families, list):
        replicated_families = []

    if protocol_status == "probe_protocol_candidate_not_certified":
        status = "probe_protocol_before_strict_claim"
        certification_claim_allowed = False
        next_step = "Stabilize the Stage-C Utility probe protocol before making operational curation claims."
    elif operational_total_effect_pass and token_caveat:
        status = "development_pass_with_token_caveat"
        certification_claim_allowed = False
        next_step = "Run certification-grade Utility with explicit token-inventory stress handling before promotion."
    elif operational_total_effect_pass and replicated_families:
        status = "operational_effect_supported_for_certification_candidate"
        certification_claim_allowed = True
        next_step = "Use as a certification candidate only if safety, contamination, and forgetting checks also hold."
    elif operational_total_effect_pass:
        status = "operational_effect_development_only"
        certification_claim_allowed = False
        next_step = "Operational curation benefit is visible, but certification still requires replicated Stage-C protocol evidence."
    elif anti_support and not selected_beats_random:
        status = "conditional_matched_support_without_total_effect"
        certification_claim_allowed = False
        next_step = "Report matched-control support as mechanism evidence only; do not make a training-use claim without Stage-A-random gain."
    else:
        status = "no_operational_utility_gain"
        certification_claim_allowed = False
        next_step = "Keep the dataset rejected or abstained for training use and inspect retained/rejected slices."

    reported_controls = [
        {
            "name": "baseline_stageA_random",
            "role": "primary_total_operational_effect",
            "selected_beats_control": selected_beats_random,
            "certification_role": "primary_utility_estimand",
        },
        {
            "name": "baseline_multi_matched_stageA_random",
            "role": "conditional_mechanism_diagnostic",
            "selected_beats_control": canonical_selected_beats,
            "certification_role": "reported_conditional_control_not_primary_gate",
        }
    ]
    if bool(protocol.get("anti_memorization_diagnostic_available")):
        reported_controls.append(
            {
                "name": "baseline_anti_memorization_matched_stageA_random",
                "role": "quality_shape_repeat_matched_mechanism_diagnostic",
                "selected_beats_control": anti_support,
                "certification_role": "reported_diagnostic_control_not_selector_objective",
                "delta_nll": anti_evidence.get("delta_nll"),
                "delta_nll_ci_low": anti_evidence.get("delta_nll_ci_low"),
                "minimum_detectable_delta_nll_95_max": anti_evidence.get("minimum_detectable_delta_nll_95_max"),
            }
        )

    return {
        "dataset": dataset,
        "status": status,
        "certification_claim_allowed": certification_claim_allowed,
        "next_step": next_step,
        "stage_c_passed": stage_c_passed,
        "coverage_passed": coverage_passed,
        "primary_operational_baseline": "baseline_stageA_random",
        "primary_operational_selected_beats_stageA_random": selected_beats_random,
        "operational_total_effect_pass": operational_total_effect_pass,
        "matched_controls_role": "conditional_mechanism_diagnostics_not_primary_gate",
        "token_exposure_caveat": token_caveat,
        "canonical_strict_status": utility.get("strict_status"),
        "canonical_selected_beats_multi_matched": canonical_selected_beats,
        "framework_implication": implication.get("status"),
        "transfer_gap_category": transfer_gap.get("category"),
        "protocol_status": protocol_status,
        "replicated_valid_power_sweep_families": replicated_families,
        "anti_memorization_diagnostic_available": bool(protocol.get("anti_memorization_diagnostic_available")),
        "anti_memorization_supports_selected": anti_support,
        "anti_memorization_evidence": anti_evidence,
        "reported_controls": reported_controls,
        "selector_policy_action": "hold",
        "utility_scope": "Stage C validation only; never selector objective",
    }


def build_report(
    readiness_report: Dict[str, Any],
    transfer_gap_report: Dict[str, Any],
    protocol_report: Dict[str, Any],
) -> Dict[str, Any]:
    readiness_datasets = readiness_report.get("datasets") or {}
    transfer_datasets = transfer_gap_report.get("datasets") or {}
    protocol_datasets = protocol_report.get("datasets") or {}
    datasets = {
        str(dataset): _strict_control_decision(
            str(dataset),
            readiness if isinstance(readiness, dict) else {},
            (transfer_datasets.get(str(dataset)) or {}) if isinstance(transfer_datasets, dict) else {},
            (protocol_datasets.get(str(dataset)) or {}) if isinstance(protocol_datasets, dict) else {},
        )
        for dataset, readiness in readiness_datasets.items()
    }
    statuses: Dict[str, int] = {}
    for payload in datasets.values():
        status = str(payload.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    certification_ready = [
        dataset for dataset, payload in datasets.items() if bool(payload.get("certification_claim_allowed"))
    ]
    return {
        "schema_version": "strict-baseline-control-report-v1",
        "profile": readiness_report.get("profile") or transfer_gap_report.get("profile") or protocol_report.get("profile"),
        "purpose": "Record Stage-C baseline/control role decisions without changing Stage-B selector objectives.",
        "framework_contract": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
            "anti_memorization_scope": "reported diagnostic control only",
            "primary_utility_estimand": "selected_vs_equal_budget_disjoint_stageA_random",
            "matched_controls_role": "conditional mechanism diagnostics, not primary certification gates",
        },
        "summary": {
            "dataset_count": len(datasets),
            "status_counts": statuses,
            "certification_claim_allowed_dataset_count": len(certification_ready),
            "certification_claim_allowed_datasets": certification_ready,
        },
        "datasets": datasets,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    lines = [
        "# Utility Baseline Control Report",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Certification-claim datasets: `{summary.get('certification_claim_allowed_dataset_count')}`",
        "- Utility scope: `Stage C validation only; never selector objective`",
        "- Anti-memorization scope: `reported diagnostic control only`",
        "",
        "## Dataset Controls",
        "",
        "| Dataset | Status | Cert claim | Operational pass | Canonical matched | Anti-mem supports | Replicated families | Selector action |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.append(
            f"| {dataset} | {payload.get('status')} | {payload.get('certification_claim_allowed')} | "
            f"{payload.get('operational_total_effect_pass')} | {payload.get('canonical_strict_status')} | "
            f"{payload.get('anti_memorization_supports_selected')} | "
            f"{payload.get('replicated_valid_power_sweep_families') or []} | {payload.get('selector_policy_action')} |"
        )
    lines.extend(["", "## Next Steps", ""])
    for dataset, payload in (report.get("datasets") or {}).items():
        controls = ", ".join(str(control.get("name")) for control in payload.get("reported_controls") or [])
        lines.extend([
            f"### {dataset}",
            "",
            f"- Next step: {payload.get('next_step')}",
            f"- Framework implication: `{payload.get('framework_implication')}`",
            f"- Reported controls: `{controls}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build strict-baseline control decision report.")
    parser.add_argument("--readiness-report", type=Path, default=CURATION_READINESS_REPORT_PATH)
    parser.add_argument("--transfer-gap-report", type=Path, default=UTILITY_TRANSFER_GAP_REPORT_PATH)
    parser.add_argument("--protocol-report", type=Path, default=STAGE_C_PROTOCOL_DECISION_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        _load_optional(args.readiness_report),
        _load_optional(args.transfer_gap_report),
        _load_optional(args.protocol_report),
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[26] strict baseline control json: {args.output}", flush=True)
    print(f"[26] strict baseline control md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
