#!/usr/bin/env python3
"""Build an operational Core audit for the LM curation framework."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_FRAMEWORK = Path("configs") / "lm_curation_operational_framework_v1.json"
DEFAULT_METRIC_SPEC = Path("configs") / "metric_spec_with_citations.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "core_operational_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_operational_audit.md"

CORE_METRIC_EXPECTATIONS: Dict[str, Dict[str, Any]] = {
    "Validity": {
        "required_metrics": ["structural_validity_gate", "structural_validity_score"],
        "stage": "Stage A",
        "required_role_terms": ["gate", "diagnostic"],
        "must_not_claim": ["semantic usefulness", "intrinsic data quality"],
    },
    "Selection Value Evidence": {
        "framework_key": "selection_value",
        "required_metrics": ["reference_quality_score"],
        "stage": "Stage B",
        "required_role_terms": ["selection_signal"],
        "must_not_claim": ["ground-truth", "intrinsic"],
    },
    "Redundancy": {
        "required_metrics": [
            "exact_duplicate_indicator",
            "shingle_near_duplicate_indicator",
            "shingle_near_duplicate_risk_score",
        ],
        "stage": "Stage A and Stage B",
        "required_role_terms": ["gate", "selection_signal"],
        "must_not_claim": ["all repeated structure is harmful"],
    },
    "Coverage": {
        "required_metrics": ["subset_coverage_retention_score", "tail_cluster_rarity_proxy"],
        "stage": "Stage B and Stage C",
        "required_role_terms": ["subset_validator", "selection_signal"],
        "must_not_claim": ["semantic/domain coverage without metadata"],
    },
    "Utility": {
        "required_metrics": ["small_lm_probe_gain_score"],
        "stage": "Stage C",
        "required_role_terms": ["subset_validator"],
        "must_not_claim": ["selector objective"],
    },
}

FORBIDDEN_STAGE_B_METRICS = {"small_lm_probe_gain_score", "predictive_utility_proxy"}


def _lower_blob(payload: Any) -> str:
    if isinstance(payload, dict):
        return " ".join(_lower_blob(v) for v in payload.values())
    if isinstance(payload, list):
        return " ".join(_lower_blob(v) for v in payload)
    return str(payload).lower()


def _metric_payload(metric_spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    metrics = metric_spec.get("metrics") or {}
    payload = metrics.get(name)
    if not isinstance(payload, dict):
        return {}
    return payload


def _metric_summary(metric_spec: Dict[str, Any], name: str) -> Dict[str, Any]:
    payload = _metric_payload(metric_spec, name)
    orthogonality = payload.get("orthogonality_contract") or {}
    return {
        "metric": name,
        "present": bool(payload),
        "role": payload.get("role"),
        "status": payload.get("status"),
        "axis": orthogonality.get("axis"),
        "claim": payload.get("claim"),
        "prohibited_signals": list(orthogonality.get("prohibited_signals") or []),
        "failure_modes": list(payload.get("failure_modes") or []),
    }


def _audit_core(
    core_name: str,
    expectation: Dict[str, Any],
    framework: Dict[str, Any],
    metric_spec: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = [_metric_summary(metric_spec, metric) for metric in expectation["required_metrics"]]
    blockers: List[str] = []
    warnings: List[str] = []

    missing = [row["metric"] for row in metrics if not row["present"]]
    if missing:
        blockers.append(f"missing_required_metrics:{','.join(missing)}")

    expected_stage = str(expectation["stage"])
    framework_key = str(expectation.get("framework_key") or core_name.lower())
    operational = framework["core_interpretation"][framework_key]
    observed_stage = str(operational.get("stage"))
    normalized_observed_stage = observed_stage.replace(" support", "").replace(" validation", "")
    if normalized_observed_stage != expected_stage:
        blockers.append(
            f"framework_stage_mismatch:{core_name}:{observed_stage}!={expected_stage}"
        )

    role_blob = " ".join(str(row.get("role") or "") for row in metrics).lower()
    for term in expectation["required_role_terms"]:
        if term.lower() not in role_blob:
            blockers.append(f"missing_role_term:{core_name}:{term}")

    core_blob = _lower_blob({"framework": operational, "metrics": metrics})
    for forbidden in expectation["must_not_claim"]:
        if forbidden.lower() in core_blob and core_name not in {"Selection Value Evidence", "Coverage", "Utility"}:
            warnings.append(f"review_claim_language:{forbidden}")

    if core_name == "Selection Value Evidence":
        boundary = str(operational.get("claim_boundary") or "").lower()
        if "not intrinsic" not in boundary and "not ground-truth" not in boundary:
            blockers.append("selection_value_boundary_does_not_reject_intrinsic_quality")
        if "no stage-a hard-reject authority" not in boundary:
            blockers.append("selection_value_boundary_missing_no_hard_reject_authority")
        quality_payload = _metric_payload(metric_spec, "reference_quality_score")
        formal = str(quality_payload.get("formal_definition") or "").lower()
        if "not a utility outcome" not in formal or "stage-b selection signal" not in formal:
            blockers.append("quality_metric_missing_stage_b_nonutility_boundary")

    if core_name == "Redundancy":
        risk_payload = _metric_payload(metric_spec, "shingle_near_duplicate_risk_score")
        risk_blob = _lower_blob(risk_payload)
        if "useful recurrence" not in risk_blob:
            blockers.append("redundancy_metric_missing_useful_recurrence_boundary")
        if "harmful_redundancy_minus_useful_recurrence" not in risk_blob:
            blockers.append("redundancy_metric_missing_harmful_vs_useful_policy")

    if core_name == "Utility":
        utility_payload = _metric_payload(metric_spec, "small_lm_probe_gain_score")
        utility_blob = _lower_blob(utility_payload)
        utility_boundary = str(operational.get("claim_boundary") or "").lower()
        if "never" not in utility_boundary or "selector objective" not in utility_boundary:
            blockers.append("utility_framework_boundary_missing_selector_prohibition")
        if "selector objective" not in utility_blob:
            blockers.append("utility_metric_missing_selector_prohibition")

    return {
        "core": core_name,
        "status": "pass" if not blockers else "fail",
        "operational_role": operational.get("role"),
        "stage": operational.get("stage"),
        "claim_boundary": operational.get("claim_boundary"),
        "required_metrics": metrics,
        "blockers": blockers,
        "warnings": warnings,
    }


def _audit_stage_b_forbidden(metric_spec: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    blockers = []
    for metric in sorted(FORBIDDEN_STAGE_B_METRICS):
        payload = _metric_payload(metric_spec, metric)
        blob = _lower_blob(payload)
        role = str(payload.get("role") or "")
        prohibited = list(((payload.get("orthogonality_contract") or {}).get("prohibited_signals") or []))
        mentions_selector_prohibition = "selector objective" in blob or "canonical selector objective" in blob
        rows.append(
            {
                "metric": metric,
                "present": bool(payload),
                "role": role,
                "mentions_selector_prohibition": mentions_selector_prohibition,
                "prohibited_signals": prohibited,
            }
        )
        if not payload:
            blockers.append(f"missing_forbidden_metric_contract:{metric}")
        if metric == "small_lm_probe_gain_score" and role != "subset_validator":
            blockers.append(f"utility_role_not_subset_validator:{role}")
        if metric == "predictive_utility_proxy" and role != "diagnostic":
            blockers.append(f"predictive_utility_proxy_role_not_diagnostic:{role}")
        if not mentions_selector_prohibition:
            blockers.append(f"missing_selector_prohibition_language:{metric}")
    return {
        "status": "pass" if not blockers else "fail",
        "forbidden_stage_b_metric_contracts": rows,
        "blockers": blockers,
    }


def build(framework_path: Path, metric_spec_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    framework = load_json(framework_path)
    metric_spec = load_json(metric_spec_path)
    core_rows = [
        _audit_core(core_name, expectation, framework, metric_spec)
        for core_name, expectation in CORE_METRIC_EXPECTATIONS.items()
    ]
    forbidden_stage_b = _audit_stage_b_forbidden(metric_spec)
    blockers = []
    for row in core_rows:
        blockers.extend(f"{row['core']}:{blocker}" for blocker in row["blockers"])
    blockers.extend(f"StageBForbidden:{blocker}" for blocker in forbidden_stage_b["blockers"])

    report = {
        "schema_version": "core-operational-audit-v1",
        "status": "core_operational_audit_passed" if not blockers else "core_operational_audit_failed",
        "source_paths": {
            "framework_contract": str(framework_path),
            "metric_spec": str(metric_spec_path),
        },
        "core_audits": core_rows,
        "stage_b_forbidden_metric_audit": forbidden_stage_b,
        "blockers": blockers,
        "interpretation": (
            "Core axes are operational curation responsibilities, not intrinsic data-quality truths. "
            "Selection Value Evidence and Redundancy may guide optional Stage-B budget allocation only as frozen pre-outcome evidence; "
            "Utility remains Stage-C validation only."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Core Operational Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        report["interpretation"],
        "",
        "## Core Axes",
        "",
        "| Core | Stage | Role | Status | Boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["core_audits"]:
        lines.append(
            "| {core} | {stage} | {role} | {status} | {boundary} |".format(
                core=row["core"],
                stage=row["stage"],
                role=row["operational_role"],
                status=row["status"],
                boundary=str(row["claim_boundary"]).replace("|", "/"),
            )
        )
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Required Next Core Work",
            "",
            "- Maintain code-domain selected-vs-rejected feature-shift diagnostics.",
            "- Separate concise useful examples/tests/bug fixes from low-information short chunks.",
            "- Keep harmful duplication and useful recurrence as distinct Redundancy reports.",
            "- Treat any missing Stage-C primary or guardrail evidence as abstention.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build operational Core audit.")
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--metric-spec", type=Path, default=DEFAULT_METRIC_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.framework, args.metric_spec, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if report["status"] == "core_operational_audit_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
