#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, filename: str):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    materialize = _load("materialize_raw_corpus_matrix", "232_materialize_raw_corpus_matrix.py")
    stages = _load("raw_corpus_matrix_stages", "233_run_raw_corpus_matrix_stages.py")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        matrix_root = root / "matrix"
        materialize.build(
            ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "release_candidates.jsonl",
            ROOT / "outputs" / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool" / "known_high_quality_raw_records.jsonl",
            ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "quarantined_candidates.jsonl",
            matrix_root,
        )
        report = stages.run(
            matrix_root,
            ROOT / "configs" / "temporal_code_curation_protocol_v1.json",
            root / "stages",
        )
    assert report["status"] == "raw_corpus_matrix_stages_materialized"
    assert report["conditions"]["clean_retain_all"]["stage_b_selection_mode"] == "retain_all"
    assert report["conditions"]["raw_mixed"]["stage_b_selection_mode"] == "budget_constrained"
    assert report["conditions"]["risk_heavy"]["stage_b_selection_mode"] == "budget_constrained"
    assert report["conditions"]["clean_retain_all"]["budget_not_selected_is_rejection"] is False
    assert report["stage_b_blinding_audit"]["forbidden_key_seen"] is False
    assert report["stage_b_blinding_audit"]["source_tier_available_to_stage_b"] is False
    assert report["frozen_input_manifest"]["stage_b_policy_sha256"]
    assert report["frozen_input_manifest"]["conditions"]["raw_mixed"]["stage_b_selected_sha256"]
    assert report["frozen_input_manifest"]["conditions"]["risk_heavy"]["stage_b_baseline_sha256"]
    assert report["training_readiness"]["stage_a_materialized"] is True
    assert report["training_readiness"]["stage_b_materialized"] is True
    assert report["training_readiness"]["primary_study_ready"] is False
    print("[raw-corpus-matrix-stages] frozen Stage A/B and blinding audit: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
