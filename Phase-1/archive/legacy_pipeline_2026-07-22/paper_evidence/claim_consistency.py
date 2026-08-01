#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT_PATH: Final = ROOT / "configs" / "paper_claim_consistency_contract_v1.json"
CODE_SUMMARY: Final = (
    OUTPUT_DIR / "validation" / "code_domain_natural_budget_current_framework_stage_c_summary_report.json"
)
CODE_PAPER: Final = OUTPUT_DIR / "validation" / "code_paper_evidence_report.json"
MATH_SUMMARY: Final = OUTPUT_DIR / "validation" / "math_domain_selector_v3_stage_c_summary_report.json"
PAPER_GATE: Final = OUTPUT_DIR / "validation" / "paper_claim_release_gate_report.json"
FINAL_TABLE: Final = OUTPUT_DIR / "validation" / "final_paper_evidence_table.json"
REPORT_PATH: Final = OUTPUT_DIR / "validation" / "paper_claim_consistency_audit_report.json"
MD_REPORT_PATH: Final = OUTPUT_DIR / "validation" / "paper_claim_consistency_audit_report.md"


@dataclass(frozen=True, slots=True)
class ClaimSources:
    contract: JsonMap
    code: JsonMap
    code_paper: JsonMap
    math: JsonMap
    paper_gate: JsonMap
    final_table: JsonMap


def _load_sources() -> ClaimSources:
    return ClaimSources(
        contract=load_json(CONTRACT_PATH),
        code=load_json(CODE_SUMMARY),
        code_paper=load_json(CODE_PAPER),
        math=load_json(MATH_SUMMARY),
        paper_gate=load_json(PAPER_GATE),
        final_table=load_json(FINAL_TABLE),
    )


def _source_sha256() -> JsonMap:
    paths = [CONTRACT_PATH, CODE_SUMMARY, CODE_PAPER, MATH_SUMMARY, PAPER_GATE, FINAL_TABLE]
    return {str(path): sha256_file(path) for path in paths}


def _table_rows(final_table: JsonMap) -> JsonMap:
    return {f"{row['domain']}::{row['arm']}": row for row in final_table["rows"]}


def _audit_code(sources: ClaimSources) -> JsonMap:
    contract = sources.contract["domain_claims"]["code"]
    raw = sources.code["arms"]["raw_full_natural"]
    curated = sources.code["arms"]["curated_v2_natural"]
    paper_nll = sources.code_paper["nll"]
    paper_evalplus = sources.code_paper["evalplus"]
    framework_compatible = bool(
        (sources.code_paper.get("framework_compatibility") or {}).get("current_artifacts_match")
    )
    external = sources.code_paper.get("external_confirmation") or {}
    external_status = external.get("status")
    external_claim = external.get("claim")
    external_bound = (
        external_status == "completed_multiseed_external_transfer_inconclusive"
        and external_claim == "external_transfer_not_demonstrated_on_frozen_livecodebench_confirmation"
    )
    decision_matches = sources.code["decision"] == contract["required_decision"]
    nll_improves = float(curated["mean_nll"]) < float(raw["mean_nll"])
    evalplus_improves = float(curated["evalplus"]["macro_pass_rate"]) > float(raw["evalplus"]["macro_pass_rate"])
    token_reduces = int(curated["packed_training_tokens"]) < int(raw["packed_training_tokens"])
    paper_matches_summary = (
        sources.code_paper["protocol_id"] == sources.code["schema_version"]
        and float(paper_nll["raw_mean_nll"]) == float(raw["mean_nll"])
        and float(paper_nll["curated_mean_nll"]) == float(curated["mean_nll"])
        and int(paper_nll["raw_packed_training_tokens"]) == int(raw["packed_training_tokens"])
        and int(paper_nll["curated_packed_training_tokens"]) == int(curated["packed_training_tokens"])
        and float(paper_evalplus["raw_macro_pass_rate"]) == float(raw["evalplus"]["macro_pass_rate"])
        and float(paper_evalplus["curated_macro_pass_rate"])
        == float(curated["evalplus"]["macro_pass_rate"])
    )
    passed = (
        framework_compatible
        and decision_matches
        and nll_improves
        and evalplus_improves
        and token_reduces
        and paper_matches_summary
        and external_bound
    )
    claim_statement = (
        contract["allowed_claim_when_current_framework_matches"]
        if passed
        else contract["historical_claim"]
    )
    return {
        "status": "pass" if passed else "blocked",
        "decision_matches_contract": decision_matches,
        "nll_improves": nll_improves,
        "evalplus_improves": evalplus_improves,
        "packed_tokens_reduce": token_reduces,
        "paper_nll_and_tokens_match_stage_c_summary": paper_matches_summary,
        "current_framework_artifacts_match": framework_compatible,
        "external_transfer_status": external_status,
        "external_transfer_claim": external_claim,
        "external_transfer_claim_bound": external_bound,
        "claim_statement": claim_statement,
    }


def _audit_math(sources: ClaimSources) -> JsonMap:
    contract = sources.contract["domain_claims"]["math"]
    decision = sources.math["decision"]
    raw = sources.math["arms"]["raw_full_natural"]
    v2 = sources.math["arms"]["curated_math_v2_natural"]
    v3 = sources.math["arms"]["curated_math_v3_natural"]
    label_matches = decision["label"] == contract["required_decision_label"]
    repair_only = bool(decision["v3_repairs_v2_failure"]) and not bool(decision["primary_success"])
    v3_repairs_v2 = float(v3["mean_nll"]) < float(v2["mean_nll"])
    v3_does_not_beat_raw = float(v3["mean_nll"]) > float(raw["mean_nll"])
    guardrail_missing = decision["benchmark_guardrail_status"] == contract["required_missing_guardrail"]
    passed = label_matches and repair_only and v3_repairs_v2 and v3_does_not_beat_raw and guardrail_missing
    return {
        "status": "abstain" if passed else "blocked",
        "decision_label_matches_contract": label_matches,
        "v3_repairs_v2": v3_repairs_v2,
        "v3_does_not_beat_raw": v3_does_not_beat_raw,
        "benchmark_guardrail_missing": guardrail_missing,
        "allowed_claim": contract["allowed_claim"],
    }


def _audit_final_table(sources: ClaimSources) -> JsonMap:
    rows = _table_rows(sources.final_table)
    domain_decisions = sources.final_table["domain_decisions"]
    required_rows = {
        "Code::curated_v2_natural": (
            "pass"
            if bool((sources.code_paper.get("framework_compatibility") or {}).get("current_artifacts_match"))
            else "historical_positive_rerun_required"
        ),
        "Math::curated_math_v2_natural": "fail",
        "Math::curated_math_v3_natural": "repair_only_abstain",
    }
    row_results = {
        key: key in rows and rows[key]["decision"] == decision
        for key, decision in required_rows.items()
    }
    code_paper = sources.code_paper
    code_row_matches_paper = (
        rows["Code::raw_full_natural"]["protocol_id"] == sources.code["schema_version"]
        and rows["Code::curated_v2_natural"]["protocol_id"] == sources.code["schema_version"]
        and float(rows["Code::curated_v2_natural"]["mean_nll"]) == float(code_paper["nll"]["curated_mean_nll"])
        and float(rows["Code::curated_v2_natural"]["evalplus_macro_pass_rate"])
        == float(code_paper["evalplus"]["curated_macro_pass_rate"])
        and float(rows["Code::raw_full_natural"]["evalplus_macro_pass_rate"])
        == float(code_paper["evalplus"]["raw_macro_pass_rate"])
    )
    decisions_match = (
        domain_decisions["Code"]
        == (
            sources.contract["domain_claims"]["code"]["status_when_current_framework_matches"]
            if bool((sources.code_paper.get("framework_compatibility") or {}).get("current_artifacts_match"))
            else sources.contract["domain_claims"]["code"]["status"]
        )
        and domain_decisions["Math"] == sources.contract["domain_claims"]["math"]["status"]
        and domain_decisions["Production"] == "blocked"
        and domain_decisions["UniversalAllDomain"] == "not_supported"
    )
    return {
        "status": "pass" if all(row_results.values()) and decisions_match and code_row_matches_paper else "blocked",
        "required_rows": row_results,
        "domain_decisions_match_contract": decisions_match,
        "code_rows_match_paper_evidence_report": code_row_matches_paper,
    }


def _audit_paper_gate(sources: ClaimSources) -> JsonMap:
    required = sources.contract["production_claim"]
    production_blocked = (
        bool(sources.paper_gate["curation_stage_framework_claim_supported"])
        and sources.paper_gate["production_deployment_claim_supported"] is required["supported"]
        and sources.paper_gate["production_status"] == required["required_status"]
    )
    return {
        "status": "pass" if production_blocked else "blocked",
        "curation_stage_framework_claim_supported": bool(sources.paper_gate["curation_stage_framework_claim_supported"]),
        "production_deployment_claim_supported": bool(sources.paper_gate["production_deployment_claim_supported"]),
        "production_status": sources.paper_gate["production_status"],
    }


def _blockers(sections: JsonMap) -> list[str]:
    return [name for name, row in sections.items() if row["status"] == "blocked"]


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Paper Claim Consistency Audit",
        "",
        f"Status: `{report['status']}`",
        f"Canonical claim: `{report['canonical_claim']['paper_claim_tier']}`",
        "",
        "## Sections",
        "",
    ]
    for name, row in report["sections"].items():
        lines.append(f"- `{name}`: `{row['status']}`")
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in report["forbidden_claims"]])
    lines.append("")
    return "\n".join(lines)


def build() -> JsonMap:
    sources = _load_sources()
    sections = {
        "code_domain": _audit_code(sources),
        "math_domain": _audit_math(sources),
        "final_evidence_table": _audit_final_table(sources),
        "paper_gate": _audit_paper_gate(sources),
    }
    blockers = _blockers(sections)
    report = {
        "schema_version": "paper-claim-consistency-audit-v1",
        "status": "paper_claim_consistency_audit_passed" if not blockers else "paper_claim_consistency_audit_blocked",
        "canonical_claim": sources.contract["canonical_claim"],
        "sections": sections,
        "blockers": blockers,
        "allowed_claims": [
            sources.contract["canonical_claim"]["scope"],
            sections["code_domain"]["claim_statement"],
            sources.contract["domain_claims"]["math"]["allowed_claim"],
        ],
        "forbidden_claims": sources.contract["canonical_claim"]["not_claimed"],
        "utility_scope": sources.contract["utility_scope"],
        "source_sha256": _source_sha256(),
        "claim_boundary": (
            "Historical Code evidence is positive but requires a rerun when implementation fingerprints do not match. "
            "Math v3 is repair-only and remains abstain. "
            "The paper claim is curation-stage and deployment-conditioned, not universal or production-ready."
        ),
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
