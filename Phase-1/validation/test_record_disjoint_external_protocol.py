#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT / "protocols" / "code_record_disjoint_confirmatory_evaluation_protocol.json"


def main() -> int:
    from external_evaluation.preflight_record_disjoint_confirmatory import preflight

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_pending_external_training"
    assert protocol["evaluation_role"] == "external_offline_validation"
    assert protocol["may_mutate_curation_output"] is False
    assert protocol["training"]["seeds"] == [101, 202, 303]
    assert protocol["training"]["comparison"] == "natural_dataset_budget_per_arm"
    assert protocol["selector_boundary"] == {
        "utility_read": False,
        "benchmark_outcomes_read": False,
        "target_token_fraction_read": False,
    }
    assert protocol["curation_input"]["integrity_gate_required"] is True
    assert protocol["curation_input"]["role"] == "confirmatory_only"
    assert {benchmark["id"] for benchmark in protocol["primary_benchmarks"]} == {
        "humanevalplus",
        "mbppplus",
        "bigcodebench_complete",
        "cruxeval_input_prediction",
        "cruxeval_output_prediction",
        "ds1000",
    }
    assert protocol["blocked_benchmarks"][0]["id"] == "livecodebench_code_generation_lite"

    report = preflight()
    assert report["status"] == "preflight_ready_for_external_training"
    assert report["pending_gates"] == []

    print("[record-disjoint-external-protocol] confirmatory natural-budget boundary: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
