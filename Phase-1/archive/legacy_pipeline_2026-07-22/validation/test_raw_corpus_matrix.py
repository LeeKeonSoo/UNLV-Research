#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "raw_corpus_matrix_report.json"


def main() -> int:
    subprocess.run([sys.executable, "231_build_raw_corpus_matrix.py"], cwd=PROJECT_DIR, check=True)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "raw_corpus_matrix_contract_frozen_materialization_pending"
    assert not report["blockers"]
    assert set(report["conditions"]) == {"clean_retain_all", "raw_mixed", "risk_heavy"}
    assert report["required_record_fields"]["source_uri"] is True
    assert report["required_record_fields"]["collected_at"] is True
    assert report["required_record_fields"]["license_family"] is True
    assert report["stage_b_blinding"]["source_tier_available_to_stage_b"] is False
    assert report["stage_b_blinding"]["known_reference_label_available_to_stage_b"] is False
    assert report["benchmark_exclusion"]["task_hash_or_registry_required"] is True
    print("[raw-corpus-matrix] provenance-rich conditions and Stage-B blinding: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
