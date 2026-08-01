#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hard_development_ablation import build_development_report


FIXTURE = ROOT / "validation" / "fixtures" / "hard_development_ablation_cases_v1.json"


def test_hard_development_ablation_compares_normal_and_hard_without_external_feedback() -> None:
    with TemporaryDirectory() as directory:
        report = build_development_report(FIXTURE, Path(directory) / "report.json")

    assert report["status"] == "hard_development_ablation_passed"
    assert report["external_evaluation_read"] is False
    assert report["runtime_forbidden_inputs"] == ["Utility", "NLL", "benchmark_outcomes", "target_retention_fraction"]
    assert {item["domain"] for item in report["scenarios"]} == {"code", "math", "general"}
    clean = [item for item in report["scenarios"] if item["clean_retention_required"]]
    assert all(item["clean_retention_passed"] for item in clean)
    assert all(item["coverage_invariant_passed"] for item in report["scenarios"])
    code = next(item for item in report["scenarios"] if item["id"] == "code-explicit-span-artifacts")
    assert set(code["hard_span_reason_codes"]) == {
        "inline_license_header_removed",
        "inline_license_comment_block_removed",
        "repeated_exact_template_span_removed",
    }
    assert code["hard_token_delta_proxy"] < 0


if __name__ == "__main__":
    test_hard_development_ablation_compares_normal_and_hard_without_external_feedback()
    print("[hard-development-ablation] Code/Math/General Normal-vs-Hard matrix: pass")
