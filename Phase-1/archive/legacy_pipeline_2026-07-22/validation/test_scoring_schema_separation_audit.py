#!/usr/bin/env python3
"""Validate scoring schema separation between Core and diagnostics."""

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
    module = _load("168_build_scoring_schema_separation_audit.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "03_score_core_metrics.py",
            ROOT / "outputs" / "validation" / "selector_utility_leakage_audit.json",
            tmp_path / "scoring_schema_separation_audit.json",
            tmp_path / "scoring_schema_separation_audit.md",
        )
    assert report["status"] == "scoring_schema_separation_audit_passed"
    assert not report["blockers"]
    assert not report["constants"]["forbidden_core_terms_in_core_constants"]
    assert report["constants"]["predictive_utility_in_diagnostic_constants"] is True
    assert report["source_contract"]["split_metric_groups_defined"] is True
    assert report["source_contract"]["grouped_scorer_api_called"] is True
    assert report["split_contract"]["predictive_utility_in_diagnostic"] is True
    assert not report["split_contract"]["extra_metric_promoted"]
    print("[scoring-schema-separation] Core metrics and diagnostic Utility proxy remain separated: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
