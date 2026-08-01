#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


REPORT_PATH = PROJECT_DIR / "outputs" / "validation" / "hf_mixed_corpus_retest_protocol_report.json"


def main() -> int:
    # Given: a frozen Qwen3-4B HF mixed-corpus protocol is expected.
    # When: the protocol audit is built.
    subprocess.run([sys.executable, "223_build_hf_mixed_corpus_retest_protocol.py"], cwd=PROJECT_DIR, check=True)
    report = load_json(REPORT_PATH)

    # Then: it must preserve provenance, forbid label leakage, and keep Utility in Stage C.
    assert report["status"] == "hf_mixed_corpus_retest_protocol_frozen"
    assert report["model"]["name"] == "Qwen/Qwen3-4B-Base"
    assert report["candidate_mixture"]["primary_mix"]["raw_like_fraction"] == 0.7
    assert report["candidate_mixture"]["primary_mix"]["known_high_quality_reference_fraction"] == 0.3
    assert report["candidate_mixture"]["stress_mix"]["raw_like_fraction"] == 0.9
    assert report["candidate_mixture"]["source_labels_preserved"] is True
    assert report["selector_leakage_controls"]["source_tier_label_available_to_stage_b"] is False
    assert report["selector_leakage_controls"]["hf_dataset_identity_available_to_stage_b"] is False
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    assert "curated_equal_token" in report["training_arms"]
    assert "raw_mixed_all_natural_budget" in report["supporting_arms"]
    assert report["required_audits"]["dataset_composition"] is True
    assert report["required_audits"]["benchmark_contamination"] is True
    assert report["hf_sources"]["raw_like"]
    assert report["hf_sources"]["known_high_quality_reference"]
    print("[hf-mixed-corpus-retest-protocol] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
