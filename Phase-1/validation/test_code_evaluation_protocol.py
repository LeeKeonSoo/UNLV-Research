#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "code_evaluation_protocol.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    primary_ids = {benchmark["id"] for benchmark in protocol["primary_benchmarks"]}

    assert protocol["evaluation_role"] == "external_offline_validation"
    assert protocol["may_mutate_curation_output"] is False
    assert protocol["training_arms"] == ["base", "raw", "curated"]
    assert protocol["training_seeds"] == [101, 202, 303]
    assert primary_ids == {
        "humanevalplus",
        "mbppplus",
        "livecodebench_code_generation_lite",
        "bigcodebench_complete",
        "cruxeval_input_prediction",
        "cruxeval_output_prediction",
        "ds1000",
    }
    assert protocol["secondary_benchmarks"][0]["id"] == "swebench_lite_fixed_agent"
    assert protocol["reporting"]["aggregate_rule"] == "report_each_benchmark_and_family_without_single_cross_metric_score"
    assert protocol["benchmark_exclusion"]["required_before_training"] is True
    print("[code-evaluation-protocol] diverse external benchmark matrix: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
