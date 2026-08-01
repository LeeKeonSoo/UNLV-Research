#!/usr/bin/env python3
"""Contract checks for EvalPlus guardrail prevalidation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    report = load_json(
        PROJECT_DIR / "outputs" / "validation" / "temporal_code_evalplus_guardrail_prevalidation.json"
    )
    assert report["status"] == "e2_prevalidated"
    assert report["decision"]["semantic_controls_executed"] is True
    assert report["decision"]["semantic_controls_pass"] is True
    assert report["decision"]["execution_support_tier"] == "E2"
    assert report["decision"]["may_enter_stage_c_guardrail"] is True
    assert report["decision"]["may_replace_primary_temporal_executable_aggregate"] is False
    assert report["environment"]["model_generated_code_executed"] is False
    assert report["environment"]["resource_module_available"] is False
    assert report["environment"]["docker_daemon_available"] is True
    assert report["environment"]["isolated_backend"] == "docker_linux"
    assert report["environment"]["isolated_image_id"]
    assert report["task_content_persisted"] is False
    assert report["confirmatory_outcomes_read"] is False
    for dataset in report["datasets"].values():
        assert dataset["executed_task_count"] == dataset["smoke_task_count"]
        assert dataset["reference_control_pass_count"] == dataset["smoke_task_count"]
        assert dataset["negative_control_rejected_count"] == dataset["smoke_task_count"]
    print("[temporal-code-evalplus] isolated Linux semantic controls establish E2 evaluator: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
