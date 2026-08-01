#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weak_development_gate import build_development_report


def main() -> int:
    fixture_path = ROOT / "validation" / "fixtures" / "weak_development_gate_cases.json"
    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "development_gate_report.json"
        report = build_development_report(fixture_path, output_path)

        assert report["status"] == "weak_development_gate_passed"
        assert report["runtime_inputs"] == ["chunk text"]
        assert report["forbidden_runtime_inputs"] == ["Utility", "NLL", "benchmark_outcomes", "target_retention_fraction"]
        assert {scenario["domain"] for scenario in report["scenarios"]} == {"code", "math", "general"}
        assert {scenario["scenario_type"] for scenario in report["scenarios"]} == {
            "clean",
            "duplicate_heavy",
            "boilerplate_heavy",
            "malformed",
        }
        assert all(scenario["coverage_invariant_passed"] for scenario in report["scenarios"])
        assert all(scenario["expected_reason_codes_observed"] for scenario in report["scenarios"])
        assert all(scenario["clean_retention_passed"] for scenario in report["scenarios"] if scenario["scenario_type"] == "clean")

        boilerplate = [scenario for scenario in report["scenarios"] if scenario["scenario_type"] == "boilerplate_heavy"]
        assert any("license_comment_only_chunk" in scenario["all_rules_removed_reason_codes"] for scenario in boilerplate)
        assert any("empty_html_shell" in scenario["all_rules_removed_reason_codes"] for scenario in boilerplate)
        assert any("explicit_web_chrome_only_chunk" in scenario["all_rules_removed_reason_codes"] for scenario in boilerplate)
        assert json.loads(output_path.read_text(encoding="utf-8")) == report

    print("[weak-development-gate] Code/Math/General rule-on/off matrix: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
