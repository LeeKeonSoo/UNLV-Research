#!/usr/bin/env python3
"""Validate Coverage/domain metadata fixture benchmark."""

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
    module = _load("167_build_coverage_domain_fixture_benchmark.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "validation" / "fixtures" / "coverage_domain_fixture_cases.json",
            tmp_path / "coverage_domain_fixture_benchmark_report.json",
            tmp_path / "coverage_domain_fixture_benchmark_report.md",
        )
    assert report["status"] == "coverage_domain_fixture_benchmark_passed"
    assert not report["blockers"]
    by_id = {row["id"]: row for row in report["cases"]}
    assert by_id["explicit_domain_balanced"]["true_domain_claim_allowed"] is True
    assert by_id["explicit_domain_balanced"]["threshold_pass"] is True
    assert by_id["explicit_domain_collapsed"]["threshold_pass"] is False
    assert set(by_id["url_domain_metadata"]["original_counts"]) == {"docs.python.org", "pypi.org"}
    assert by_id["source_bucket_fallback_only"]["support"]["support_scope"] == "source_bucket_fallback"
    assert by_id["source_bucket_fallback_only"]["true_domain_claim_allowed"] is False
    assert by_id["mixed_domain_and_source_bucket"]["support"]["support_scope"] == "mixed_domain_and_source_bucket"
    assert by_id["mixed_domain_and_source_bucket"]["true_domain_claim_allowed"] is False
    print("[coverage-domain-fixture] explicit metadata, fallback scope, and collapse checks: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
