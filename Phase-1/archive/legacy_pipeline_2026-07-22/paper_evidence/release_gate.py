#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, save_json, sha256_file


DEFAULT_CORE_BEHAVIOR = OUTPUT_DIR / "validation" / "core_behavior_audit_v2.json"
DEFAULT_SELECTOR_LEAKAGE = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_V2_CONFIRMATORY = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_CANONICAL_GUARDRAIL = (
    Path("validation") / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json"
)
DEFAULT_TARGET_SIZE = (
    Path("validation") / "frozen_contracts" / "redundancy_target_size_qwen3_4b_development_report.json"
)
DEFAULT_CODE_PAPER_EVIDENCE = OUTPUT_DIR / "validation" / "code_paper_evidence_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "paper_claim_release_gate_report.json"


def _load(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _source(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _v2_has_unresolved_confirmatory_evidence(v2: Dict[str, Any]) -> bool:
    if v2.get("status") != "v2_confirmatory_decision_passed":
        return True
    summary = v2.get("summary") if isinstance(v2.get("summary"), dict) else {}
    if summary.get("blockers"):
        return True
    nll_gate = summary.get("nll_gate") if isinstance(summary.get("nll_gate"), dict) else {}
    if nll_gate.get("status") != "passed":
        return True
    if summary.get("training_runs_completed") != summary.get("expected_training_runs"):
        return True
    if summary.get("heldout_nll_results_completed") != summary.get("expected_heldout_nll_results"):
        return True
    guardrails = summary.get("stage_c_guardrails") if isinstance(summary.get("stage_c_guardrails"), dict) else {}
    return any(
        not isinstance(row, dict) or row.get("evidence_state") != "passed" or row.get("passed") is not True
        for row in guardrails.values()
    )


def build(
    core_behavior_path: Path,
    selector_leakage_path: Path,
    v2_confirmatory_path: Path,
    canonical_guardrail_path: Path,
    target_size_path: Path,
    code_paper_evidence_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    reports = {
        "core_behavior": _load(core_behavior_path),
        "selector_utility_leakage": _load(selector_leakage_path),
        "v2_confirmatory": _load(v2_confirmatory_path),
        "canonical_guardrail": _load(canonical_guardrail_path),
        "target_size": _load(target_size_path),
        "code_paper_evidence": _load(code_paper_evidence_path),
    }
    sources = {
        "core_behavior": _source(core_behavior_path),
        "selector_utility_leakage": _source(selector_leakage_path),
        "v2_confirmatory": _source(v2_confirmatory_path),
        "canonical_guardrail": _source(canonical_guardrail_path),
        "target_size": _source(target_size_path),
        "code_paper_evidence": _source(code_paper_evidence_path),
    }
    blockers: List[str] = []
    production_blockers: List[str] = []
    for name, source in sources.items():
        if not source["exists"]:
            blockers.append(f"missing_required_report:{name}")

    core = reports["core_behavior"] or {}
    if core.get("status") != "core_behavior_audit_development_checks_passed":
        blockers.append("core_behavior_audit_not_current_behavior_check_pass")
    decision = core.get("decision") if isinstance(core.get("decision"), dict) else {}
    if decision.get("release_claim_supported") is not True:
        production_blockers.append("production_core_validity_not_supported")
    if decision.get("core_metric_validity_fully_proven") is not True:
        if "production_core_validity_not_supported" not in production_blockers:
            production_blockers.append("production_core_validity_not_supported")

    leakage = reports["selector_utility_leakage"] or {}
    if leakage.get("status") != "selector_utility_leakage_audit_passed":
        blockers.append("selector_utility_leakage_audit_not_passed")

    v2 = reports["v2_confirmatory"] or {}
    if v2.get("status") != "v2_confirmatory_decision_passed":
        blockers.append(f"v2_confirmatory_not_release_pass:{v2.get('status')}")
    if _v2_has_unresolved_confirmatory_evidence(v2):
        blockers.append("v2_confirmatory_has_abstain_missing_or_incomplete_evidence")

    canonical = reports["canonical_guardrail"] or {}
    if canonical.get("release_decision") != "release_supported":
        blockers.append(f"canonical_guardrail_release_not_supported:{canonical.get('release_decision')}")

    target = reports["target_size"] or {}
    if target.get("status") != "target_size_development_passed":
        blockers.append(f"target_size_not_full_development_pass:{target.get('status')}")
    target_guardrails = target.get("guardrail_status") if isinstance(target.get("guardrail_status"), dict) else {}
    if target_guardrails.get("missing_guardrails"):
        blockers.append("target_size_missing_required_guardrails")
    if target_guardrails.get("failed_guardrails"):
        blockers.append("target_size_failed_required_guardrails")
    if target_guardrails.get("release_decision") != "release_supported":
        blockers.append(f"target_size_release_not_supported:{target_guardrails.get('release_decision')}")

    code_paper = reports["code_paper_evidence"] or {}
    compatibility = code_paper.get("framework_compatibility") or {}
    if compatibility.get("current_artifacts_match") is not True:
        blockers.append("current_framework_stage_c_rerun_required")
    external = code_paper.get("external_confirmation") or {}
    if external.get("status") != "completed_multiseed_external_transfer_inconclusive":
        blockers.append("external_transfer_confirmation_missing_or_unbounded")
    if external.get("claim") != "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation":
        blockers.append("external_transfer_claim_boundary_not_frozen")

    supported = not blockers
    production_supported = supported and not production_blockers
    report = {
        "schema_version": "paper-claim-release-gate-v1",
        "status": (
            "paper_curation_stage_claim_gate_passed"
            if supported
            else "paper_curation_stage_claim_gate_blocked"
        ),
        "supported": supported,
        "paper_claim_tier": "curation_stage_research_framework",
        "curation_stage_framework_claim_supported": supported,
        "production_deployment_claim_supported": production_supported,
        "blockers": blockers,
        "production_blockers": production_blockers,
        "production_status": (
            "production_deployment_claim_gate_passed"
            if production_supported
            else "production_deployment_claim_gate_blocked"
        ),
        "sources": sources,
        "claim_boundary": (
            "This gate supports the paper claim that the project is a curation-stage "
            "framework for language-model training data. It does not certify a "
            "production deployment or universal data-quality detector."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hard-fail paper/release claim gate.")
    parser.add_argument("--core-behavior", type=Path, default=DEFAULT_CORE_BEHAVIOR)
    parser.add_argument("--selector-leakage", type=Path, default=DEFAULT_SELECTOR_LEAKAGE)
    parser.add_argument("--v2-confirmatory", type=Path, default=DEFAULT_V2_CONFIRMATORY)
    parser.add_argument("--canonical-guardrail", type=Path, default=DEFAULT_CANONICAL_GUARDRAIL)
    parser.add_argument("--target-size", type=Path, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--code-paper-evidence", type=Path, default=DEFAULT_CODE_PAPER_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.core_behavior,
        args.selector_leakage,
        args.v2_confirmatory,
        args.canonical_guardrail,
        args.target_size,
        args.code_paper_evidence,
        args.output,
    )
    print(json.dumps({"status": report["status"], "supported": report["supported"], "blockers": report["blockers"]}, indent=2))
    return 0 if report["supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
