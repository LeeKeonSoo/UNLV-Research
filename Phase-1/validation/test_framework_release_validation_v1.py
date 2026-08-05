#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_release_validation import build_release_validation


def main() -> int:
    report = build_release_validation(ROOT)
    frozen_path = ROOT / "validation" / "frozen_contracts" / "framework_release_validation_v1.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    assert report == frozen
    assert report["schema_version"] == "framework-release-validation-v1"
    assert report["implementation_integrity"] == "passed"
    assert report["framework_release"] == "blocked"
    assert report["core_behavior"]["passed"] is True
    assert set(report["core_behavior"]["cores"]) == {
        "validity",
        "redundancy",
        "quality",
        "coverage",
    }

    gates = {gate["id"]: gate for gate in report["integrity_gates"]}
    assert set(gates) == {
        "foundation_hash_chain",
        "kernel_hash_tamper_detection",
        "threshold_provenance_completeness",
        "stage_core_authority",
        "runtime_forbidden_input",
        "provider_no_direct_deletion",
        "profile_no_uncalibrated_or_unpromoted_release",
        "hard_retained_set_monotonicity",
        "curated_output_equivalence",
    }
    assert all(gate["passed"] for gate in gates.values())
    assert gates["kernel_hash_tamper_detection"]["observed_reason_code"] == (
        "runtime_bridge_kernel_identity_mismatch"
    )
    assert gates["runtime_forbidden_input"]["observed_reason_code"] == (
        "stage_runtime_forbidden_input:stage_b:benchmark_outcomes"
    )
    assert gates["curated_output_equivalence"]["observed_sha256"] == (
        gates["curated_output_equivalence"]["expected_sha256"]
    )

    assert report["release_blockers"] == [
        "profile_contains_unpromoted_policy",
        "profile_operating_points_uncalibrated",
        "coverage.representative_guard:candidate",
        "quality.explicit_nonpayload:candidate",
        "quality.teacher_panel_v2:candidate",
        "redundancy.exact_text_family:development_passed",
        "redundancy.symmetric_near_duplicate_candidate:candidate",
        "validity.interpretable_text:candidate",
    ]
    print("[framework-release-validation-v1] integrity pass, scientific release blocked: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
