#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "code_six_benchmark_execution_qwen3_4b_natural_v1.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["evaluation_role"] == "external_offline_validation"
    assert protocol["may_mutate_curation_output"] is False
    assert protocol["training"]["base_model"] == "Qwen/Qwen3-4B-Base"
    assert protocol["training"]["seeds"] == [11, 23, 37]
    assert protocol["training"]["budget"] == "natural_dataset_budget_per_arm"
    assert protocol["primary_benchmarks"] == [
        "humanevalplus",
        "mbppplus",
        "livecodebench_v6_code_generation_lite",
        "bigcodebench_complete",
        "ds1000",
        "ojbench",
    ]
    assert protocol["secondary_benchmarks"] == ["swebench_lite_fixed_agent"]
    assert protocol["selector_boundary"]["benchmark_outcomes_read"] is False
    assert protocol["selector_boundary"]["utility_read"] is False
    print("[code-six-benchmark-protocol] execution contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
