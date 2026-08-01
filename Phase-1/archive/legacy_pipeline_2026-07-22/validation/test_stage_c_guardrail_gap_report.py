#!/usr/bin/env python3
"""Validate Stage-C guardrail gap report semantics."""

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
    module = _load("171_build_stage_c_guardrail_gap_report.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json",
            ROOT / "outputs" / "validation" / "code_domain_v2_evalplus_confirmatory_guardrail_report.json",
            ROOT / "outputs" / "validation" / "code_domain_v2_general_task_confirmatory_guardrail_report.json",
            ROOT / "outputs" / "validation" / "code_domain_v2_general_text_confirmatory_guardrail_report.json",
            tmp_path / "stage_c_guardrail_gap_report.json",
            tmp_path / "stage_c_guardrail_gap_report.md",
        )
    assert report["decision_report"]["nll_gate_status"] == "passed"
    assert report["status"] == "stage_c_guardrail_gaps_closed"
    assert "evalplus_confirmatory" not in report["incomplete_guardrails"]
    assert "general_task_retention" not in report["incomplete_guardrails"]
    assert "general_text_nll_retention" not in report["incomplete_guardrails"]
    assert report["guardrails"]["evalplus_confirmatory"]["status"] == "evalplus_confirmatory_guardrail_passed"
    assert report["guardrails"]["general_task_retention"]["status"] == "general_task_confirmatory_guardrail_passed"
    print("[stage-c-guardrail-gap] required guardrails are closed: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
