#!/usr/bin/env python3
"""Validate the code-domain QLoRA runner can consume the confirmatory protocol."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json

RUNNER_PATH = ROOT / "141_run_code_domain_development_qlora.py"
PROTOCOL_PATH = ROOT / "configs" / "code_domain_confirmatory_protocol_qwen3_4b_v1.json"
DECISION_SCRIPT_PATH = ROOT / "151_build_code_domain_confirmatory_decision_report.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runner = _load_module(RUNNER_PATH, "code_domain_runner_contract")
    decision = _load_module(DECISION_SCRIPT_PATH, "code_domain_confirmatory_decision_contract")
    protocol = load_json(PROTOCOL_PATH)

    assert runner._stage_label(protocol) == "confirmatory"
    assert runner._training_seeds(protocol) == [101, 131, 163, 197, 239]
    assert runner._eval_blocks_name(protocol) == "confirmatory_code_nll_heldout.pt"
    heldout = runner._heldout_jsonl_path(protocol)
    assert heldout.exists(), heldout
    assert str(heldout).endswith("confirmatory_code_nll_heldout.jsonl")
    assert runner._qlora_completed_status(protocol) == "confirmatory_qlora_completed"

    recipe = runner._training_recipe(protocol)
    assert recipe["optimizer_steps"] == 8
    assert recipe["same_seed_set_for_every_arm"] is True

    tmp_report = Path(tempfile.gettempdir()) / "_contract_tmp_code_domain_confirmatory_decision_report.json"
    report = decision.build(
        PROTOCOL_PATH,
        ROOT / "outputs" / "validation" / "code_domain_confirmatory_protocol_qwen3_4b_report.json",
        ROOT / "outputs" / "code_domain_confirmatory_qwen3_4b_v1",
        tmp_report,
        ROOT / "outputs" / "validation" / "_missing_evalplus_confirmatory_guardrail_report.json",
        ROOT / "outputs" / "validation" / "_missing_general_text_confirmatory_guardrail_report.json",
        ROOT / "outputs" / "validation" / "_missing_general_task_confirmatory_guardrail_report.json",
    )
    assert report["status"] in {
        "confirmatory_decision_incomplete",
        "confirmatory_decision_abstain_missing_required_guardrails",
        "confirmatory_decision_reject_primary_margin_failure",
        "confirmatory_decision_reject_raw_direction_failure",
        "confirmatory_decision_reject_guardrail_failure",
        "confirmatory_decision_passed",
    }
    assert report["summary"]["expected_training_runs"] == 20
    assert report["summary"]["expected_heldout_nll_results"] == 21
    assert report["summary"]["primary_success_rule"] == protocol["primary_success_rule"]
    assert report["utility_scope"] == "Stage C validation only; never selector objective"
    tmp_report.unlink(missing_ok=True)
    print("[code-domain-confirmatory-runner] protocol-aware QLoRA runner contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
