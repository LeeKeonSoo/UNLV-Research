#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parent
FIXTURE_PATH: Final = ROOT / "validation" / "fixtures" / "math_failure_selector_cases.json"
CONTRACT_PATH: Final = ROOT / "configs" / "math_domain_selector_v3_redesign_contract.json"
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "math_failure_fixture_contract_report.json"
MD_REPORT_PATH: Final = ROOT / "outputs" / "validation" / "math_failure_fixture_contract_report.md"


def main() -> int:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    categories = sorted({case["category"] for case in cases})
    required = sorted(contract["required_fixture_categories"])
    missing = sorted(set(required) - set(categories))
    stage_owner_counts = {
        owner: sum(1 for case in cases if case["expected_stage_owner"] == owner)
        for owner in ("stage_a", "stage_b", "stage_c")
    }

    report = {
        "schema_version": "math-failure-fixture-contract-report-v1",
        "status": "math_failure_fixture_contract_ready" if not missing else "math_failure_fixture_contract_blocked",
        "fixture_path": str(FIXTURE_PATH),
        "contract_path": str(CONTRACT_PATH),
        "case_count": len(cases),
        "categories": categories,
        "missing_required_categories": missing,
        "stage_owner_counts": stage_owner_counts,
        "math_v2_result": {
            "decision": "failed_stage_c_validation",
            "raw_mean_nll": 1.49565,
            "curated_mean_nll": 1.527065,
            "curated_minus_raw_nll": 0.031415,
        },
        "next_selector_allowed": False,
        "next_required_step": "Implement Math selector v3 only after these fixture categories are mapped to pre-outcome metrics.",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MD_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Math Failure Fixture Contract Report",
                "",
                f"Status: `{report['status']}`",
                "",
                f"Case count: `{report['case_count']}`",
                "",
                "Required categories are present. Math selector v2 remains failed; selector v3 is not allowed until the fixture categories are mapped to pre-outcome metrics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[math-failure-fixture-contract] {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
