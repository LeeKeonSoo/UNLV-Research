#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    protocol = load_json(PROJECT_DIR / "configs" / "paper_multidomain_benchmark_protocol_v1.json")

    assert protocol["status"] == "frozen_design_no_training_outcomes"
    assert "single quality score" in protocol["novelty_hypothesis"][0]
    assert protocol["claim_boundary"]["expanded_claim_requires"].startswith("At least one non-code domain")

    arms = set(protocol["training_arms"])
    assert {
        "base_no_update",
        "raw_random_equal_token",
        "stageA_random_equal_token",
        "curated_stageB_equal_token",
        "known_high_quality_reference_equal_token",
    }.issubset(arms)

    accounting = set(protocol["candidate_pool_design"]["required_size_accounting"])
    assert {
        "raw_record_count",
        "raw_token_proxy",
        "stageB_selected_chunk_count",
        "stageB_selected_token_proxy",
        "training_payload_sha256",
    }.issubset(accounting)

    domains = {domain["domain"]: domain for domain in protocol["domains"]}
    assert {"code", "math", "general_text_instruction"}.issubset(set(domains))
    assert any("SWE-bench" in metric for metric in domains["code"]["primary_stage_c_metrics"])
    assert "GSM8K accuracy" in domains["math"]["primary_stage_c_metrics"]
    assert "MATH accuracy" in domains["math"]["primary_stage_c_metrics"]

    forbidden = " ".join(protocol["forbidden_uses"])
    assert "benchmark outcome inside Stage-B selector objectives" in forbidden
    assert "equal-token training budgets" in forbidden
    assert "pre/post curation record, chunk, and token counts" in forbidden

    requirements = set(protocol["paper_reporting_requirements"])
    assert "dataset composition table with raw, Stage-0, Stage-A, and Stage-B sizes" in requirements
    assert "coverage and redundancy shift figure" in requirements

    print("[paper-multidomain-benchmark-protocol] frozen design contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
