#!/usr/bin/env python3
"""Regression checks for frozen temporal-code Stage-B proxy validation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    module = importlib.import_module("76_validate_temporal_code_stage_b_proxies")
    fixtures = load_json(PROJECT_DIR / "validation" / "fixtures" / "temporal_code_stage_b_proxy_cases.json")
    report = module.validate(fixtures)
    assert report["summary"]["failed_count"] == 0, report
    assert "Utility" in report["forbidden_evidence"], report
    assert report["utility_scope"] == "Stage C validation only; never selector objective", report
    print("[temporal-code-stage-b-proxies] automated direction checks: pass")
    print("[temporal-code-stage-b-proxies] Utility and benchmark outcomes excluded: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
