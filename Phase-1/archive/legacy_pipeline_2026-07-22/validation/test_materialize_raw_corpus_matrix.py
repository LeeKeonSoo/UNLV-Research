#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "232_materialize_raw_corpus_matrix.py"
    spec = importlib.util.spec_from_file_location("materialize_raw_corpus_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        report = module.build(
            ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "release_candidates.jsonl",
            ROOT / "outputs" / "temporal_code_training_freeze_v1" / "known_high_quality_reference_pool" / "known_high_quality_raw_records.jsonl",
            ROOT / "outputs" / "temporal_code_collection" / "stage0_code_domain_v2_combined" / "train" / "quarantined_candidates.jsonl",
            Path(tmp),
        )
    assert report["status"] == "raw_corpus_matrix_materialized"
    assert not report["blockers"]
    assert report["matrix_config_sha256"]
    assert report["conditions"]["clean_retain_all"]["eligible_record_count"] == 100
    assert report["conditions"]["raw_mixed"]["source_tier_counts"] == {"known_high_quality_reference": 75, "raw_like": 175}
    assert report["conditions"]["risk_heavy"]["source_tier_counts"] == {"known_high_quality_reference": 20, "raw_like": 180}
    assert report["conditions"]["risk_heavy"]["quarantined_record_count"] == 3
    assert report["stage_b_blinding"]["source_tier_available_to_stage_b"] is False
    assert report["stage_b_blinding"]["known_reference_label_available_to_stage_b"] is False
    assert report["provenance_audit"]["missing_required_field_count"] == 0
    assert report["training_readiness"]["stage_a_materialized"] is False
    assert report["training_readiness"]["stage_b_materialized"] is False
    assert report["training_readiness"]["primary_study_ready"] is False
    print("[materialize-raw-corpus-matrix] real raw/reference provenance and quarantine conditions: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
