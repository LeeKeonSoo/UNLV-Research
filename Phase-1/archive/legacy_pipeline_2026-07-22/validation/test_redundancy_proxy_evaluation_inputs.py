#!/usr/bin/env python3
"""Validate frozen mechanism and retention inputs for the redundancy proxy."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json, sha256_file


CONFIG = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_evaluation_inputs_v1.json"
)
REPORT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_evaluation_inputs_freeze_report.json"
)


def _content_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def main() -> int:
    config = load_json(CONFIG)
    report = load_json(REPORT)
    assert config["status"] == "frozen_before_proxy_training_outcomes"
    assert report["status"] == "redundancy_proxy_evaluation_inputs_frozen"
    assert report["config_sha256"] == sha256_file(CONFIG)
    assert not config["blockers"]
    assert not report["blockers"]
    assert config["comparison_arms"] == [
        "base_no_update",
        "binary_current_equal_budget",
        "log_count_equal_budget",
        "stageA_random_common_disjoint_equal_budget",
    ]
    assert config["training_seeds"] == [11, 23, 37]

    mechanism = config["mechanism_diagnostic"]
    assert mechanism["status"] == "template_saturation_mechanism_precheck_passed"
    assert all(mechanism["checks"].values())
    assert "match_count>=2" in mechanism["definition"]
    binary = mechanism["exact_stream_exposure"]["binary_current_equal_budget"]
    candidate = mechanism["exact_stream_exposure"]["log_count_equal_budget"]
    assert candidate["high_saturation_token_share_count_ge_2"] <= binary[
        "high_saturation_token_share_count_ge_2"
    ]
    assert candidate["repository_count"] >= binary["repository_count"]

    general_text = config["general_text_retention"]
    block = general_text["blocks"]
    path = Path(block["path"])
    assert path.exists()
    assert sha256_file(path) == block["file_sha256"]
    tensor = load_file(path)["input_ids"]
    assert tensor.dtype == torch.int32
    assert tuple(tensor.shape) == (496, 1024)
    assert tensor.numel() == 507904
    assert _content_sha256(tensor) == block["tensor_content_sha256"]
    assert general_text["maximum_allowed_mean_nll_increase"] == 0.01

    general_tasks = config["general_task_retention"]
    assert general_tasks["version"] == "0.4.12"
    assert general_tasks["num_fewshot"] == 0
    assert general_tasks["limit"] is None
    assert set(general_tasks["tasks"]) == {
        "hellaswag",
        "arc_challenge",
        "piqa",
        "winogrande",
    }
    expected_examples = {
        "hellaswag": 10042,
        "arc_challenge": 299,
        "piqa": 1838,
        "winogrande": 1267,
    }
    for task, expected in expected_examples.items():
        row = general_tasks["tasks"][task]
        assert row["validation_examples"] == expected
        assert sha256_file(Path(row["validation_split"]["path"])) == row[
            "validation_split"
        ]["sha256"]
        for artifact in row["task_implementation"]:
            assert sha256_file(Path(artifact["path"])) == artifact["sha256"]

    code = config["code_retention"]
    assert code["version"] == "0.3.1"
    assert code["development_task_count"] == 284
    assert code["suite_counts"] == {"HumanEval+": 90, "MBPP+": 194}
    assert code["execution_support_tier"] == "E2"
    assert code["task_content_training_authorized"] is False
    for artifact in code["dataset_cache"].values():
        assert sha256_file(Path(artifact["path"])) == artifact["sha256"]

    decision = config["decision_contract"]
    assert decision["all_guardrails_mandatory"] is True
    assert decision["missing_evidence_action"] == "abstain"
    assert decision["mechanism_precheck_must_pass_before_training"] is True
    assert config["confirmatory_outcomes_read"] is False
    assert config["utility_scope"].startswith("Stage C validation only")
    print("[redundancy-proxy-evaluation-inputs] mechanism and retention inputs frozen: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
