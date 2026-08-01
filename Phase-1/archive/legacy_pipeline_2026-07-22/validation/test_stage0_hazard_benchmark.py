#!/usr/bin/env python3
"""Validate the Stage-0 hazard benchmark contract."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load("166_build_stage0_hazard_benchmark.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "validation" / "fixtures" / "stage0_hazard_benchmark_cases.json",
            tmp_path / "stage0_hazard_benchmark_report.json",
            tmp_path / "stage0_hazard_benchmark_report.md",
        )
    assert report["status"] == "stage0_hazard_fixture_benchmark_passed"
    assert not report["blockers"]
    assert report["summary"]["case_count"] >= 10
    by_id = {row["id"]: row for row in report["cases"]}
    assert by_id["clean_allowed_html"]["eligible"] is True
    assert by_id["pii_email"]["eligible"] is False
    assert "pii_detected" in by_id["pii_phone_general"]["reasons"]
    assert by_id["code_numeric_false_positive_suppressed"]["eligible"] is True
    assert "secret_detected" in by_id["secret_api_key"]["reasons"]
    assert "benchmark_contamination" in by_id["benchmark_contamination_humaneval"]["reasons"]
    assert "poisoning_suspected" in by_id["poisoning_instruction"]["reasons"]
    assert "rights_restricted" in by_id["rights_restricted"]["reasons"]
    assert by_id["missing_source_lineage"]["eligible"] is False
    assert "missing_provenance_source_uri" in by_id["missing_source_lineage"]["reasons"]
    assert by_id["repository_code_preserves_operators"]["eligible"] is True
    print("[stage0-hazard-benchmark] labeled hazard and false-positive fixtures: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
