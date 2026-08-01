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
    path = ROOT / "199_build_raw_vs_curated_benchmark_readiness.py"
    spec = importlib.util.spec_from_file_location("raw_vs_curated_benchmark_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        report = module.build(
            tmp_path / "raw_vs_curated_benchmark_readiness_report.json",
            tmp_path / "raw_vs_curated_benchmark_readiness_report.md",
        )

    assert report["status"] == "code_domain_ready_next_domains_pending"
    assert report["claim_contract"]["selector_rule"] == (
        "Benchmarks and Utility are Stage-C only and never Stage-B selector inputs."
    )
    assert report["code_domain"]["payloads_ready"] is True
    assert report["code_domain"]["equal_packed_tokens"] is True
    assert report["code_domain"]["packed_token_values"] == [325632]
    assert report["code_domain"]["curated_vs_stageA_random_mean_nll_reduction"] > 0.003

    arms = report["code_domain"]["arms"]
    for arm in (
        "raw_random_equal_budget",
        "stageA_random_equal_budget",
        "curated_v2_equal_budget",
        "known_high_quality_equal_budget",
    ):
        assert arms[arm]["source_jsonl_exists"] is True
        assert arms[arm]["token_block_exists"] is True
        assert arms[arm]["packed_tokens"] == 325632

    domain_status = {item["domain"]: item["status"] for item in report["domain_status"]}
    assert domain_status["code"] == "payloads_ready_swebench_pending"
    assert domain_status["math"] == "stage_c_protocol_frozen_training_pending"
    assert domain_status["general_text_instruction"] == "acquisition_required"
    assert "SWE-bench Lite" in " ".join(report["domain_status"][0]["remaining"])
    assert "GSM8K" in " ".join(report["next_block_actions"])
    assert "math fine-tuning arms" in " ".join(report["next_block_actions"])
    assert report["sources"]["math_block4_report"]["exists"] is True
    assert report["sources"]["math_equal_token_report"]["exists"] is True
    assert report["sources"]["math_stage_c_protocol"]["exists"] is True
    print("[raw-vs-curated-benchmark-readiness] code ready; next domains pending: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
