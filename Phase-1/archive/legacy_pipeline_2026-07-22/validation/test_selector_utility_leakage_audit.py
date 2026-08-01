#!/usr/bin/env python3
"""Validate selector Utility leakage audit."""

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
    module = _load("164_build_selector_utility_leakage_audit.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "policy" / "subsets.py",
            ROOT / "ingestion" / "code_selection.py",
            ROOT / "outputs" / "temporal_code_collection" / "stage_b_code_domain_v2" / "train_scored_full_selector.jsonl",
            tmp_path / "selector_utility_leakage_audit.json",
            tmp_path / "selector_utility_leakage_audit.md",
            None,
        )
    assert report["status"] == "selector_utility_leakage_audit_passed"
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert not report["blockers"]
    assert report["selector_files"]["policy_subsets"]["functions"]["_axis_scores"]["function_found"] is True
    assert report["selector_files"]["policy_subsets"]["functions"]["_axis_scores"]["forbidden_terms_found"] == []
    assert not report["stage_b_evidence_scan"]["forbidden_terms_seen"]
    assert report["stage_b_evidence_scan"]["truncated"] is False
    print("[selector-utility-leakage] Stage-B selector consumes no Utility surrogate fields: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
