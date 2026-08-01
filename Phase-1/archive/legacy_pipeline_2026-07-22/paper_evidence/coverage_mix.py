#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from data_eval_common import load_json, save_json, sha256_file


ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_PATH: Final = ROOT / "configs" / "coverage_domain_mix_contract_v1.json"
REPORT_PATH: Final = ROOT / "outputs" / "validation" / "coverage_domain_mix_audit_report.json"
MD_REPORT_PATH: Final = ROOT / "outputs" / "validation" / "coverage_domain_mix_audit_report.md"

JsonMap = dict[str, Any]


@dataclass(frozen=True, slots=True)
class CoverageMixInputs:
    contract: JsonMap
    coverage_fixture: JsonMap
    domain_composition: JsonMap
    domain_mix_contract: JsonMap


def _input_path(contract: JsonMap, key: str) -> Path:
    return ROOT / str(contract["required_inputs"][key])


def _load_inputs() -> CoverageMixInputs:
    contract = load_json(CONTRACT_PATH)
    return CoverageMixInputs(
        contract=contract,
        coverage_fixture=load_json(_input_path(contract, "coverage_fixture_report")),
        domain_composition=load_json(_input_path(contract, "domain_composition_report")),
        domain_mix_contract=load_json(_input_path(contract, "domain_mix_contract")),
    )


def _input_statuses(inputs: CoverageMixInputs) -> JsonMap:
    return {
        "coverage_fixture_report": inputs.coverage_fixture["status"],
        "domain_composition_report": inputs.domain_composition["status"],
        "domain_mix_contract": inputs.domain_mix_contract["status"],
    }


def _status_blockers(inputs: CoverageMixInputs, statuses: JsonMap) -> list[str]:
    required = inputs.contract["required_statuses"]
    return [
        f"{name}_expected_{expected}_got_{statuses.get(name)}"
        for name, expected in required.items()
        if statuses.get(name) != expected
    ]


def _share_drift(composition: JsonMap) -> dict[str, float]:
    drift = composition["mixes"]["curated_minus_raw_share_drift"]
    return {str(domain): float(value) for domain, value in drift.items()}


def _max_abs(values: dict[str, float]) -> float:
    return max((abs(value) for value in values.values()), default=0.0)


def _source_sha256(inputs: CoverageMixInputs) -> JsonMap:
    paths = {
        "coverage_domain_mix_contract": CONTRACT_PATH,
        "coverage_fixture_report": _input_path(inputs.contract, "coverage_fixture_report"),
        "domain_composition_report": _input_path(inputs.contract, "domain_composition_report"),
        "domain_mix_contract": _input_path(inputs.contract, "domain_mix_contract"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Coverage Domain-Mix Audit",
        "",
        f"Status: `{report['status']}`",
        f"Coverage role: `{report['coverage_role']}`",
        f"Current scope: `{report['coverage_scope']['current_scope']}`",
        "",
        "## Domain Share Drift",
        "",
        "| Domain | Curated minus raw share |",
        "| --- | ---: |",
    ]
    for domain, drift in report["domain_share_drift"].items():
        lines.append(f"| {domain} | {drift:.6f} |")
    lines.extend(
        [
            "",
            "## Target Mix",
            "",
            f"- Status: `{report['target_mix']['status']}`",
            f"- Target-mix claim allowed: `{report['target_mix']['target_mix_claim_allowed']}`",
            f"- Observed-composition claim allowed: `{report['target_mix']['observed_composition_claim_allowed']}`",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def build() -> JsonMap:
    inputs = _load_inputs()
    statuses = _input_statuses(inputs)
    blockers = _status_blockers(inputs, statuses)
    drift = _share_drift(inputs.domain_composition)
    target_status = str(inputs.domain_composition["target_domain_mix_status"])
    target_mix_claim_allowed = bool(inputs.contract["target_mix_claim_allowed"]) and target_status == "declared"
    observed_claim_allowed = bool(inputs.contract["observed_composition_claim_allowed"])
    report = {
        "schema_version": "coverage-domain-mix-audit-v1",
        "status": (
            "coverage_domain_mix_audit_passed_with_scope_boundary"
            if not blockers
            else "coverage_domain_mix_audit_blocked"
        ),
        "coverage_core_definition": inputs.contract["coverage_core_definition"],
        "coverage_role": inputs.contract["coverage_role"],
        "input_statuses": statuses,
        "coverage_fixture_summary": inputs.coverage_fixture["summary"],
        "coverage_scope": {
            "current_scope": inputs.domain_composition["contract_mode"],
            "true_domain_claim_policy": "requires_explicit_metadata_or_declared_contract",
            "fixture_support_scope_counts": inputs.coverage_fixture["summary"]["support_scope_counts"],
        },
        "target_mix": {
            "status": target_status,
            "target_mix_claim_allowed": target_mix_claim_allowed,
            "observed_composition_claim_allowed": observed_claim_allowed,
        },
        "domain_share_drift": drift,
        "max_abs_domain_share_drift": _max_abs(drift),
        "blockers": blockers,
        "forbidden_claims": inputs.contract["forbidden_claims"],
        "stage_boundary": inputs.contract["stage_boundary"],
        "utility_scope": inputs.contract["utility_scope"],
        "source_sha256": _source_sha256(inputs),
        "claim_boundary": (
            "Coverage can report observed domain-arm composition drift for the current paper evidence. "
            "Because no target mix is declared, it cannot claim target-mix satisfaction, Utility, "
            "intrinsic quality, or a universal domain ratio."
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
