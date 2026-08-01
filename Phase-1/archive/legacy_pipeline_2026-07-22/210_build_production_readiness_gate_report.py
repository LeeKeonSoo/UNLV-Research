#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, PROJECT_DIR, load_json, save_json


JsonMap = dict[str, Any]

DEFAULT_STAGE0_HAZARD = OUTPUT_DIR / "validation" / "stage0_hazard_benchmark_report.json"
DEFAULT_STAGE0_RISK = OUTPUT_DIR / "validation" / "stage0_risk_boundary_report.json"
DEFAULT_UTILITY_LEAKAGE = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_FINAL_EVIDENCE = OUTPUT_DIR / "validation" / "final_paper_evidence_table.json"
DEFAULT_DISPOSITION_AUDIT = OUTPUT_DIR / "validation" / "record_disposition_audit_report.json"
DEFAULT_GATE_SPEC = PROJECT_DIR / "docs" / "production_readiness_gate_spec.md"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "production_readiness_gate_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "production_readiness_gate_report.md"


def _load_existing(path: Path) -> JsonMap:
    return load_json(path) if path.exists() else {}


def _rows_by_key(report: JsonMap) -> dict[tuple[str, str], JsonMap]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return {}
    keyed: dict[tuple[str, str], JsonMap] = {}
    for row in rows:
        if isinstance(row, dict):
            keyed[(str(row.get("domain")), str(row.get("arm")))] = row
    return keyed


def _code_passes(evidence: JsonMap) -> bool:
    rows = _rows_by_key(evidence)
    raw = rows.get(("Code", "raw_full_natural"))
    curated = rows.get(("Code", "curated_v2_natural"))
    if raw is None or curated is None:
        return False
    return (
        curated.get("decision") == "pass"
        and float(curated.get("mean_nll", 0.0)) < float(raw.get("mean_nll", 0.0))
        and float(curated.get("evalplus_macro_pass_rate", 0.0))
        > float(raw.get("evalplus_macro_pass_rate", 0.0))
    )


def _math_failure_visible(evidence: JsonMap) -> bool:
    rows = _rows_by_key(evidence)
    raw = rows.get(("Math", "raw_full_natural"))
    curated = rows.get(("Math", "curated_math_v2_natural"))
    if raw is None or curated is None:
        return False
    return curated.get("decision") == "fail" and float(curated.get("mean_nll", 0.0)) > float(raw.get("mean_nll", 0.0))


def build(output_path: Path, md_output_path: Path) -> JsonMap:
    stage0_hazard = _load_existing(DEFAULT_STAGE0_HAZARD)
    stage0_risk = _load_existing(DEFAULT_STAGE0_RISK)
    utility = _load_existing(DEFAULT_UTILITY_LEAKAGE)
    evidence = _load_existing(DEFAULT_FINAL_EVIDENCE)
    disposition = _load_existing(DEFAULT_DISPOSITION_AUDIT)

    stage0_hazard_passed = stage0_hazard.get("status") == "stage0_hazard_fixture_benchmark_passed"
    stage0_boundary_scoped = stage0_risk.get("status") == "stage0_risk_boundary_scoped_not_production_ready"
    utility_passed = utility.get("status") == "selector_utility_leakage_audit_passed"
    code_passed = _code_passes(evidence)
    math_visible = _math_failure_visible(evidence)
    disposition_passed = disposition.get("status") == "record_disposition_audit_passed"
    gate_spec_exists = DEFAULT_GATE_SPEC.exists()

    blockers = [
        name
        for name, passed in {
            "stage0_hazard_fixture_not_passed": stage0_hazard_passed,
            "stage0_boundary_missing_or_overclaimed": stage0_boundary_scoped,
            "utility_leakage_audit_not_passed": utility_passed,
            "code_positive_case_missing": code_passed,
            "math_failure_boundary_missing": math_visible,
            "record_disposition_audit_missing": disposition_passed,
            "production_gate_spec_missing": gate_spec_exists,
        }.items()
        if not passed
    ]
    r3_blockers = [
        "external_detector_validation_missing",
        "legal_license_clearance_missing",
        "benchmark_contamination_registry_incomplete",
        "monitoring_and_rollback_missing",
        "real_operational_pilot_missing",
    ]
    report = {
        "schema_version": "production-readiness-gate-report-v1",
        "status": "production_gate_prototype_passed" if not blockers else "production_gate_prototype_blocked",
        "readiness_level": "R1_production_gate_prototype" if not blockers else "R0_research_claim_only",
        "paper_claim_ready": not blockers,
        "production_certified": False,
        "stage0_hazard_fixture_passed": stage0_hazard_passed,
        "stage0_boundary_scoped_not_production_ready": stage0_boundary_scoped,
        "utility_leakage_audit_passed": utility_passed,
        "code_positive_case_passed": code_passed,
        "math_failure_boundary_visible": math_visible,
        "record_disposition_audit_passed": disposition_passed,
        "production_gate_spec_exists": gate_spec_exists,
        "blockers": blockers,
        "r3_blockers": r3_blockers,
        "forbidden_claims": [
            "production_certification",
            "universal_data_quality_detection",
            "all_domain_improvement",
            "stage_b_utility_objective",
        ],
        "claim_boundary": (
            "R1 means the production-readiness gates are specified and exercised by "
            "prototype fixtures/reports. It does not certify production deployment."
        ),
        "sources": {
            "stage0_hazard": str(DEFAULT_STAGE0_HAZARD),
            "stage0_risk": str(DEFAULT_STAGE0_RISK),
            "utility_leakage": str(DEFAULT_UTILITY_LEAKAGE),
            "final_evidence": str(DEFAULT_FINAL_EVIDENCE),
            "record_disposition_audit": str(DEFAULT_DISPOSITION_AUDIT),
            "production_gate_spec": str(DEFAULT_GATE_SPEC),
        },
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Production Readiness Gate Report",
        "",
        f"Status: `{report['status']}`",
        f"Readiness level: `{report['readiness_level']}`",
        f"Paper claim ready: `{report['paper_claim_ready']}`",
        f"Production certified: `{report['production_certified']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Prototype Gates",
        "",
        "| Gate | Passed |",
        "| --- | --- |",
        f"| Stage-0 hazard fixture | `{report['stage0_hazard_fixture_passed']}` |",
        f"| Stage-0 scoped boundary | `{report['stage0_boundary_scoped_not_production_ready']}` |",
        f"| Utility leakage audit | `{report['utility_leakage_audit_passed']}` |",
        f"| Code positive case | `{report['code_positive_case_passed']}` |",
        f"| Math failure boundary | `{report['math_failure_boundary_visible']}` |",
        f"| Record disposition audit | `{report['record_disposition_audit_passed']}` |",
        f"| Production gate spec | `{report['production_gate_spec_exists']}` |",
        "",
        "## R1 Blockers",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["blockers"]] or ["- None"])
    lines.extend(["", "## R3 Production Certification Blockers", ""])
    lines.extend([f"- `{item}`" for item in report["r3_blockers"]])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in report["forbidden_claims"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build production-readiness gate prototype report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.output, args.md_output)
    print({"status": report["status"], "readiness_level": report["readiness_level"], "blockers": report["blockers"]})
    return 0 if report["status"] == "production_gate_prototype_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
