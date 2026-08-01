#!/usr/bin/env python3
"""Regression checks for evidence-based temporal-code collection readiness."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _module():
    path = PROJECT_DIR / "67_build_temporal_code_collection_readiness.py"
    spec = importlib.util.spec_from_file_location("temporal_code_collection_readiness", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def main() -> int:
    module = _module()
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        protocol = _write(
            root / "protocol.json",
            {
                "protocol_name": "fixture",
                "benchmark_quarantine": {"checks": ["repository_identity"]},
            },
        )
        benchmark_seed = _write(root / "benchmark.json", {"entries": []})
        discovery = _write(
            root / "discovery.json",
            {"summary": {"candidate_count": 10, "metadata_enrichment_candidate_count": 10}},
        )
        enrichment = _write(
            root / "enrichment.json",
            {"summary": {"repository_count": 3, "eligible_for_reproducibility_probe_count": 3}},
        )
        reproducibility = _write(
            root / "reproducibility.json",
            {"summary": {"repository_count": 3, "eligible_for_quarantine_review_count": 3}},
        )
        audit = _write(root / "audit.json", {"summary": {"collection_gate_pass_count": 2}})
        verification = _write(
            root / "verification.json",
            {
                "dry_run": False,
                "summary": {"verified_bundle_count": 2, "failed_or_unverified_bundle_count": 0},
            },
        )
        stage0 = _write(root / "stage0.json", {"summary": {"release_candidate_records": 5}})
        report = module.build(
            protocol,
            benchmark_seed,
            discovery,
            enrichment,
            reproducibility,
            root / "report.json",
            [],
            audit,
            verification,
            stage0,
        )
        assert report["status"] == "smoke_feasibility_validated_broad_manifest_not_ready", report
        assert report["evidence"]["smoke_feasibility"]["validated"] is True
        assert report["blockers"] == ["repository_enrichment_coverage_incomplete"], report

        enrichment = _write(
            root / "enrichment-full.json",
            {"summary": {"repository_count": 10, "eligible_for_reproducibility_probe_count": 8}},
        )
        reproducibility = _write(
            root / "reproducibility-full.json",
            {"summary": {"repository_count": 8, "eligible_for_quarantine_review_count": 8}},
        )
        ready = module.build(
            protocol,
            benchmark_seed,
            discovery,
            enrichment,
            reproducibility,
            root / "ready.json",
            [],
            audit,
            verification,
            stage0,
        )
        assert ready["status"] == "ready_to_freeze_repository_manifest", ready
        assert ready["blockers"] == []
        frozen_manifest = _write(
            root / "frozen.json",
            {"summary": {"frozen_repository_count": 8}},
        )
        frozen = module.build(
            protocol,
            benchmark_seed,
            discovery,
            enrichment,
            reproducibility,
            root / "frozen-report.json",
            [],
            audit,
            verification,
            stage0,
            frozen_manifest,
        )
        assert frozen["status"] == "broad_repository_manifest_frozen", frozen
        assert frozen["frozen_repository_count"] == 8
    print("[temporal-code-collection-readiness] smoke evidence and broad coverage gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
