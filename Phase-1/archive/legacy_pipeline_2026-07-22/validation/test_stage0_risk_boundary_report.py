#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_script():
    path = ROOT / "193_build_stage0_risk_boundary_report.py"
    spec = importlib.util.spec_from_file_location("stage0_risk_boundary_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            ROOT / "outputs" / "validation" / "stage0_hazard_benchmark_report.json",
            ROOT / "outputs" / "validation" / "stage0_detector_validation_report.json",
            ROOT / "outputs" / "validation" / "stage0_detector_heldout_benchmark_report.json",
            ROOT / "outputs" / "validation" / "real_corpus_stage0_coverage_audit.json",
            tmp_path / "stage0_risk_boundary_report.json",
            tmp_path / "stage0_risk_boundary_report.md",
        )
    assert report["status"] == "stage0_risk_boundary_scoped_not_production_ready"
    assert report["claim_decision"]["production_detector_claim_supported"] is False
    assert report["claim_decision"]["legal_rights_clearance_claim_supported"] is False
    assert report["claim_decision"]["benchmark_contamination_exhaustive_claim_supported"] is False
    assert report["real_corpus_stage0"]["quarantined_candidate_count"] == 6
    assert report["real_corpus_stage0"]["release_candidate_count"] == 312
    for axis in ("pii_detected", "secret_detected", "benchmark_contamination", "poisoning_suspected", "rights_allowed"):
        assert axis in report["risk_axes"]
        assert report["risk_axes"][axis]["development_fixture_recall"] == 1.0
        assert report["risk_axes"][axis]["heldout_fixture_recall"] == 1.0
    assert "external_public_detector_benchmark_missing" in report["remaining_evidence_gaps"]
    print("[stage0-risk-boundary] scoped Stage-0 risk claims and blockers are explicit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
