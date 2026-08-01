#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parent
VALIDATION_DIR: Final = ROOT / "outputs" / "validation"
REPORT_PATH: Final = VALIDATION_DIR / "stage0_release_blocker_report.json"
MD_REPORT_PATH: Final = VALIDATION_DIR / "stage0_release_blocker_report.md"
SOURCE_REPORTS: Final = [
    "stage0_hazard_benchmark_report.json",
    "stage0_detector_validation_report.json",
    "stage0_detector_heldout_benchmark_report.json",
    "stage0_risk_boundary_report.json",
    "real_corpus_stage0_coverage_audit.json",
    "production_readiness_gate_report.json",
]
PRODUCTION_BLOCKERS: Final = [
    "external_pii_detector_validation_missing",
    "external_secret_detector_validation_missing",
    "license_compliance_validation_missing",
    "benchmark_contamination_validation_missing",
    "adversarial_poisoning_validation_missing",
]


def main() -> int:
    sources = {}
    for name in SOURCE_REPORTS:
        path = VALIDATION_DIR / name
        data = json.loads(path.read_text(encoding="utf-8"))
        sources[name] = {
            "status": data.get("status"),
            "claim_boundary": data.get("claim_boundary", ""),
        }

    report = {
        "schema_version": "stage0-release-blocker-report-v1",
        "status": "stage0_release_blocked_production_guardrails",
        "development_evidence_present": True,
        "production_release_allowed": False,
        "production_gate_prototype_status": sources["production_readiness_gate_report.json"]["status"],
        "production_gate_prototype_ready": sources["production_readiness_gate_report.json"]["status"]
        == "production_gate_prototype_passed",
        "source_reports": sources,
        "production_blockers": PRODUCTION_BLOCKERS,
        "claim_boundary": "Stage-0 has project fixture and heldout precheck evidence, but this is not production detector validation for PII, secrets, licensing, contamination, or poisoning.",
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Stage-0 Release Blocker Report",
                "",
                "Status: `stage0_release_blocked_production_guardrails`",
                "",
                "Development evidence is present, but production release is blocked.",
                "",
                f"Production-gate prototype: `{report['production_gate_prototype_status']}`",
                "",
                "Production blockers:",
                "",
                *[f"- `{blocker}`" for blocker in PRODUCTION_BLOCKERS],
                "",
                "Claim boundary: Stage-0 evidence is not production detector validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("[stage0-release-blocker] stage0_release_blocked_production_guardrails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
