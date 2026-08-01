#!/usr/bin/env python3
"""Build the canonical redundancy guardrail decision after Stage-C checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    ROOT / "configs" / "temporal_code_redundancy_canonical_guardrails_qwen25_0p5b_v1.json"
)
DEFAULT_PROXY_DECISION = (
    ROOT / "validation" / "frozen_contracts" / "redundancy_proxy_decision_report.json"
)
DEFAULT_NLL = (
    ROOT
    / "outputs"
    / "redundancy_saturation_proxy_qwen25_0p5b_v1"
    / "nll_evaluation"
    / "nll_summary.json"
)
DEFAULT_GENERAL_TASK = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_canonical_general_task_guardrail_report.json"
)
DEFAULT_EVALPLUS = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_canonical_evalplus_guardrail_report.json"
)
DEFAULT_TARGET_SIZE = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_target_size_qwen3_4b_development_report.json"
)
DEFAULT_V2_CONFIRMATORY = (
    ROOT / "outputs" / "validation" / "code_domain_v2_confirmatory_decision_report.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_canonical_guardrail_decision_report.json"
)


def build(
    contract_path: Path,
    proxy_decision_path: Path,
    nll_path: Path,
    general_task_path: Path,
    evalplus_path: Path,
    target_size_path: Path,
    v2_confirmatory_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    contract = load_json(contract_path)
    proxy_decision = load_json(proxy_decision_path)
    nll = load_json(nll_path)
    general_task = load_json(general_task_path)
    evalplus = load_json(evalplus_path)
    target_size = load_json(target_size_path)
    v2_confirmatory = load_json(v2_confirmatory_path)
    blockers = []
    if contract.get("status") != "frozen_before_canonical_guardrail_outcomes":
        blockers.append("canonical_guardrail_contract_not_frozen")
    if proxy_decision.get("canonical_control") != "binary_current":
        blockers.append("canonical_control_not_binary_current")
    if proxy_decision["curation_effect"].get("passed") is not True:
        blockers.append("curation_vs_random_target_nll_missing")
    if nll["general_text_retention"].get("passed") is not True:
        blockers.append("general_text_retention_failed_or_missing")
    if general_task.get("status") != "general_task_guardrail_passed":
        blockers.append(f"general_task_not_passed:{general_task.get('status')}")
    if evalplus.get("status") != "evalplus_development_guardrail_passed":
        blockers.append(f"evalplus_not_passed:{evalplus.get('status')}")
    target_size_guardrails = target_size.get("guardrail_status") or {}
    if target_size.get("status") != "target_size_development_passed":
        blockers.append(f"target_size_not_passed:{target_size.get('status')}")
    if target_size_guardrails.get("release_decision") != "release_supported":
        blockers.append(f"target_size_release_not_supported:{target_size_guardrails.get('release_decision')}")
    if v2_confirmatory.get("status") != "v2_confirmatory_decision_passed":
        blockers.append(f"v2_confirmatory_not_passed:{v2_confirmatory.get('status')}")

    all_development_checks_passed = not blockers
    release_supported = all_development_checks_passed
    report = {
        "schema_version": "redundancy-canonical-guardrail-decision-report-v1",
        "status": (
            "canonical_qwen25_0p5b_development_guardrails_passed"
            if all_development_checks_passed
            else "canonical_qwen25_0p5b_development_guardrails_blocked"
        ),
        "canonical_selector_path": "binary_current_equal_budget",
        "rejected_candidate_path": "log_count_equal_budget",
        "source_sha256": {
            str(contract_path): sha256_file(contract_path),
            str(proxy_decision_path): sha256_file(proxy_decision_path),
            str(nll_path): sha256_file(nll_path),
            str(general_task_path): sha256_file(general_task_path),
            str(evalplus_path): sha256_file(evalplus_path),
            str(target_size_path): sha256_file(target_size_path),
            str(v2_confirmatory_path): sha256_file(v2_confirmatory_path),
        },
        "evidence": {
            "target_code_nll_curation_vs_stageA_random": proxy_decision[
                "curation_effect"
            ],
            "general_text_retention": nll["general_text_retention"],
            "general_task_retention": general_task["comparisons_vs_base"][
                "binary_current_equal_budget"
            ],
            "evalplus_development_retention": evalplus["comparisons_vs_base"][
                "binary_current_equal_budget"
            ],
        },
        "development_conclusion": (
            "The canonical binary recurrence path improved target heldout code NLL over "
            "the common disjoint Stage-A random baseline and passed frozen Stage-C "
            "general-text, general-task, and EvalPlus development guardrails."
        ),
        "target_size_release_decision": target_size_guardrails.get("release_decision"),
        "v2_confirmatory_decision": v2_confirmatory.get("status"),
        "release_decision": "release_supported" if release_supported else "abstain_not_a_production_release",
        "release_blockers": blockers,
        "forbidden_interpretations": [
            "claiming log_count is promoted",
            "claiming production-ready framework release from this 0.5B development proxy alone",
            "using Stage-C Utility or guardrail outcomes inside Stage-B selection",
        ],
        "blockers": blockers,
        "confirmatory_outcomes_read": True,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Development evidence for the canonical binary recurrence path on Qwen2.5-0.5B. "
            "It supports the canonical binary_current path after target-size and confirmatory "
            "guardrails are observed; paper-level release still depends on the separate Core claim gate."
        ),
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--proxy-decision", type=Path, default=DEFAULT_PROXY_DECISION)
    parser.add_argument("--nll", type=Path, default=DEFAULT_NLL)
    parser.add_argument("--general-task", type=Path, default=DEFAULT_GENERAL_TASK)
    parser.add_argument("--evalplus", type=Path, default=DEFAULT_EVALPLUS)
    parser.add_argument("--target-size", type=Path, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--v2-confirmatory", type=Path, default=DEFAULT_V2_CONFIRMATORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.contract,
        args.proxy_decision,
        args.nll,
        args.general_task,
        args.evalplus,
        args.target_size,
        args.v2_confirmatory,
        args.output,
    )
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
