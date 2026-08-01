#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "code_7benchmark_pretraining_eligible_v3_execution.json"


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    benchmarks = protocol["primary_benchmarks"]
    benchmark_ids = [benchmark["id"] for benchmark in benchmarks]

    assert protocol["evaluation_role"] == "external_offline_validation"
    assert protocol["may_mutate_curation_output"] is False
    assert protocol["training"]["arms"] == ["base_no_update", "stage_a_release_natural_v3", "curated_natural_v3"]
    assert protocol["training"]["seeds"] == [101, 202, 303]
    assert protocol["training"]["comparison"] == "natural_dataset_budget_per_arm"
    materialized = protocol["training"]["materialized_inputs"]
    assert materialized["stage_a_release_natural_v3"] == {"materialized_tokens": 6963200, "optimizer_steps": 425}
    assert materialized["curated_natural_v3"] == {"materialized_tokens": 6815744, "optimizer_steps": 416}
    assert protocol["selector_boundary"] == {
        "utility_read": False,
        "benchmark_outcomes_read": False,
        "target_token_fraction_read": False,
    }
    assert benchmark_ids == [
        "humanevalplus",
        "mbppplus",
        "livecodebench_code_generation_lite",
        "bigcodebench_complete",
        "cruxeval_input_prediction",
        "cruxeval_output_prediction",
        "ds1000",
    ]
    assert all(benchmark["metric"] == "pass_at_1" for benchmark in benchmarks)
    crux_contracts = [
        benchmark["generation_file_contract"]
        for benchmark in benchmarks
        if benchmark["id"].startswith("cruxeval_")
    ]
    assert len(crux_contracts) == 2
    assert all("sample_0 through sample_799" in contract for contract in crux_contracts)
    temporal = protocol["temporal_declaration"]
    assert temporal["model_pretraining_cutoff"] is None
    assert temporal["raw_corpus_snapshot_end"] is None
    assert temporal["required_before_livecodebench_generation"] is True
    assert protocol["status"] == "blocked_pending_v3_training_and_temporal_declarations"
    print("[code-7benchmark-pretraining-eligible-v3] external execution contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
