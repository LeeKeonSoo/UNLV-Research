#!/usr/bin/env python3
"""Validate Stage-B Redundancy saturation ablations."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.code_selection import _structural_saturation_risk  # noqa: E402


def _load():
    path = ROOT / "178_run_redundancy_saturation_ablations.py"
    spec = importlib.util.spec_from_file_location("redundancy_saturation_ablations", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    assert _structural_saturation_risk(1, "binary_current") == 0.85
    assert _structural_saturation_risk(4, "binary_current") == 0.85
    assert _structural_saturation_risk(4, "exp_tau_1") > _structural_saturation_risk(1, "exp_tau_1")
    assert _structural_saturation_risk(4, "exp_tau_2") > _structural_saturation_risk(1, "exp_tau_2")
    assert _structural_saturation_risk(4, "log_count") > _structural_saturation_risk(1, "log_count")

    source = (
        ROOT
        / "outputs"
        / "temporal_code_collection"
        / "stage_a_code_domain_v2_balanced"
        / "train"
        / "stage_a_pass.jsonl"
    )
    if not source.exists():
        print("[redundancy-saturation-ablations] formula checks pass; corpus run skipped")
        return 0
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report = module.build(
            source,
            ROOT / "configs" / "temporal_code_curation_protocol_v1.json",
            ROOT / "configs" / "temporal_code_redundancy_saturation_ablation_v1.json",
            root / "report.json",
            root / "report.md",
        )
    assert set(report["arms"]) == {"binary_current", "exp_tau_1", "exp_tau_2", "log_count"}
    assert report["arms"]["binary_current"]["jaccard_with_current"] == 1.0
    assert report["utility_scope"].startswith("Stage C only")
    print("[redundancy-saturation-ablations] monotonic formulas and outcome-free corpus shifts: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
