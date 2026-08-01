#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from data_eval_common import load_json, save_json, sha256_file


ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_PATH: Final = ROOT / "configs" / "stage_b_policy_contract_v1.json"
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "stage_b_policy_contract_audit_report.json"
MD_REPORT_PATH: Final = ROOT / "outputs" / "validation" / "stage_b_policy_contract_audit_report.md"

JsonMap = dict[str, Any]


@dataclass(frozen=True, slots=True)
class StageBPolicyInputs:
    contract: JsonMap
    framework: JsonMap
    disposition: JsonMap
    leakage: JsonMap
    coverage: JsonMap


def _path(contract: JsonMap, key: str) -> Path:
    return ROOT / str(contract["required_reports"][key])


def _load_inputs() -> StageBPolicyInputs:
    contract = load_json(CONTRACT_PATH)
    return StageBPolicyInputs(
        contract=contract,
        framework=load_json(_path(contract, "operational_framework")),
        disposition=load_json(_path(contract, "record_disposition_audit")),
        leakage=load_json(_path(contract, "selector_utility_leakage_audit")),
        coverage=load_json(_path(contract, "coverage_domain_mix_audit")),
    )


def _statuses(inputs: StageBPolicyInputs) -> JsonMap:
    return {
        "operational_framework": inputs.framework["status"],
        "record_disposition_audit": inputs.disposition["status"],
        "selector_utility_leakage_audit": inputs.leakage["status"],
        "coverage_domain_mix_audit": inputs.coverage["status"],
    }


def _blockers(inputs: StageBPolicyInputs, statuses: JsonMap) -> list[str]:
    required = inputs.contract["required_statuses"]
    status_blockers = [
        f"{name}_expected_{expected}_got_{statuses.get(name)}"
        for name, expected in required.items()
        if statuses.get(name) != expected
    ]
    framework_invariants = set(inputs.framework["disposition_contract"]["invariants"])
    invariant_blockers = [
        f"missing_framework_invariant:{item}"
        for item in inputs.contract["required_invariants"]
        if item not in framework_invariants
    ]
    semantic_checks = {
        "retain_all_not_valid": bool(inputs.disposition["retain_all_is_valid"]),
        "budget_not_selected_marked_rejection": not bool(inputs.disposition["budget_not_selected_is_rejection"]),
        "unexpected_stage_b_evidence_keys_present": not inputs.leakage["stage_b_evidence_scan"][
            "unexpected_stage_b_evidence_keys"
        ],
    }
    semantic_blockers = [name for name, passed in semantic_checks.items() if not passed]
    return [*status_blockers, *invariant_blockers, *semantic_blockers]


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Stage-B Policy Contract Audit",
        "",
        f"Status: `{report['status']}`",
        f"Role: `{report['stage_b_role']}`",
        "",
        "## Activation",
        "",
        f"- Binding budget required: `{report['activation']['binding_budget_required']}`",
        f"- No-binding-budget action: `{report['activation']['no_binding_budget_action']}`",
        "",
        "## Semantics",
        "",
        f"- `retain_all_is_valid`: `{report['disposition_semantics']['retain_all_is_valid']}`",
        f"- `budget_not_selected_is_rejection`: `{report['disposition_semantics']['budget_not_selected_is_rejection']}`",
        f"- `budget_not_selected_is_low_quality`: `{report['disposition_semantics']['budget_not_selected_is_low_quality']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["blockers"]] or ["- None"])
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def build() -> JsonMap:
    inputs = _load_inputs()
    statuses = _statuses(inputs)
    blockers = _blockers(inputs, statuses)
    stage_b_contract = inputs.framework["stage_contract"]["stage_b"]
    report = {
        "schema_version": "stage-b-policy-contract-audit-v1",
        "status": "stage_b_policy_contract_audit_passed" if not blockers else "stage_b_policy_contract_audit_blocked",
        "stage_b_role": inputs.contract["canonical_role"],
        "input_statuses": statuses,
        "activation": {
            "rule": inputs.contract["activation_rule"],
            "framework_activation": stage_b_contract["activation"],
            "binding_budget_required": "smaller than the full curated pool" in stage_b_contract["activation"],
            "no_binding_budget_action": inputs.contract["no_binding_budget_action"],
        },
        "disposition_semantics": {
            "retain_all_is_valid": bool(inputs.disposition["retain_all_is_valid"]),
            "budget_not_selected_is_rejection": bool(inputs.disposition["budget_not_selected_is_rejection"]),
            "budget_not_selected_is_low_quality": False,
            "observed_training_budget_dispositions": inputs.disposition["observed_training_budget_dispositions"],
        },
        "selector_boundary": {
            "utility_leakage_status": inputs.leakage["status"],
            "unexpected_stage_b_evidence_keys": inputs.leakage["stage_b_evidence_scan"][
                "unexpected_stage_b_evidence_keys"
            ],
            "forbidden_terms_seen": inputs.leakage["stage_b_evidence_scan"]["forbidden_terms_seen"],
        },
        "coverage_boundary": {
            "observed_composition_claim_allowed": bool(
                inputs.coverage["target_mix"]["observed_composition_claim_allowed"]
            ),
            "target_mix_claim_allowed": bool(inputs.coverage["target_mix"]["target_mix_claim_allowed"]),
            "coverage_role": inputs.coverage["coverage_role"],
        },
        "blockers": blockers,
        "forbidden_stage_b_inputs": inputs.contract["forbidden_stage_b_inputs"],
        "forbidden_claims": inputs.contract["forbidden_claims"],
        "utility_scope": inputs.contract["utility_scope"],
        "source_sha256": {
            "stage_b_policy_contract": sha256_file(CONTRACT_PATH),
            "operational_framework": sha256_file(_path(inputs.contract, "operational_framework")),
            "record_disposition_audit": sha256_file(_path(inputs.contract, "record_disposition_audit")),
            "selector_utility_leakage_audit": sha256_file(_path(inputs.contract, "selector_utility_leakage_audit")),
            "coverage_domain_mix_audit": sha256_file(_path(inputs.contract, "coverage_domain_mix_audit")),
        },
        "claim_boundary": (
            "Stage B is an optional budget allocator over retained Stage-A survivors. "
            "It does not reject usable records, does not require shrinkage, and does not consume Utility."
        ),
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
