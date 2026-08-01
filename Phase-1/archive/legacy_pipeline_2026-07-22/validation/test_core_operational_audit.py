#!/usr/bin/env python3
"""Validate the operational Core audit contract."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "161_build_core_operational_audit.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_builder():
    spec = importlib.util.spec_from_file_location("core_operational_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_builder()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report_path = tmp_path / "core_operational_audit.json"
        report = module.build(
            ROOT / "configs" / "lm_curation_operational_framework_v1.json",
            ROOT / "configs" / "metric_spec_with_citations.json",
            report_path,
            tmp_path / "core_operational_audit.md",
        )
        saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "core_operational_audit_passed"
    assert report["utility_scope"] == "Stage C validation only; never selector objective"

    by_core = {row["core"]: row for row in saved["core_audits"]}
    selection_value = by_core["Selection Value Evidence"]
    assert selection_value["operational_role"] == "observable_pre_outcome_selection_evidence"
    assert selection_value["stage"] == "Stage B"
    assert "not intrinsic" in selection_value["claim_boundary"]
    assert "no Stage-A hard-reject authority" in selection_value["claim_boundary"]
    assert by_core["Redundancy"]["status"] == "pass"
    assert by_core["Utility"]["stage"] == "Stage C"

    forbidden = saved["stage_b_forbidden_metric_audit"]
    assert forbidden["status"] == "pass"
    proxy_rows = {
        row["metric"]: row
        for row in forbidden["forbidden_stage_b_metric_contracts"]
    }
    assert proxy_rows["predictive_utility_proxy"]["role"] == "diagnostic"
    assert proxy_rows["small_lm_probe_gain_score"]["role"] == "subset_validator"

    print("[core-operational-audit] Core axes remain operational framework responsibilities: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
