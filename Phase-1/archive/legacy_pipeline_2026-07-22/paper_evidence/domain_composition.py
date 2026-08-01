#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_eval_common import load_json, save_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "domain_mix_contract_v1.json"
FINAL_TABLE_PATH = ROOT / "outputs" / "validation" / "final_paper_evidence_table.json"
REPORT_PATH = ROOT / "outputs" / "validation" / "domain_composition_audit_report.json"
MD_REPORT_PATH = ROOT / "outputs" / "validation" / "domain_composition_audit_report.md"

JsonMap = dict[str, Any]


@dataclass(frozen=True, slots=True)
class DomainSpec:
    domain: str
    source_report: Path
    raw_arm: str
    curated_arm: str
    required_curated_decision: str


def _domain_specs(contract: JsonMap) -> list[DomainSpec]:
    specs = []
    for row in contract["domains"]:
        specs.append(
            DomainSpec(
                domain=str(row["domain"]),
                source_report=ROOT / str(row["source_report"]),
                raw_arm=str(row["raw_arm"]),
                curated_arm=str(row["curated_arm"]),
                required_curated_decision=str(row["required_curated_decision"]),
            )
        )
    return specs


def _table_rows(final_table: JsonMap) -> JsonMap:
    return {f"{row['domain']}::{row['arm']}": row for row in final_table["rows"]}


def _arm_snapshot(arms: JsonMap, arm_name: str) -> JsonMap:
    arm = arms[arm_name]
    return {
        "records": arm.get("records"),
        "token_proxy_count": arm.get("token_proxy_count"),
        "packed_training_tokens": int(arm["packed_training_tokens"]),
        "optimizer_steps": arm.get("optimizer_steps"),
        "mean_nll": arm.get("mean_nll"),
    }


def _fraction_reduced(raw_value: int | None, curated_value: int | None) -> float | None:
    if raw_value is None or curated_value is None or raw_value == 0:
        return None
    return 1.0 - (float(curated_value) / float(raw_value))


def _mix(rows: list[JsonMap], arm_key: str) -> JsonMap:
    token_counts = {row["domain"]: int(row[arm_key]["packed_training_tokens"]) for row in rows}
    total = sum(token_counts.values())
    shares = {
        domain: (float(tokens) / float(total) if total else 0.0)
        for domain, tokens in token_counts.items()
    }
    return {"total_packed_training_tokens": total, "tokens": token_counts, "shares": shares}


def _domain_row(spec: DomainSpec, final_rows: JsonMap) -> JsonMap:
    source = load_json(spec.source_report)
    raw = _arm_snapshot(source["arms"], spec.raw_arm)
    curated = _arm_snapshot(source["arms"], spec.curated_arm)
    final_key = f"{spec.domain}::{spec.curated_arm}"
    curated_decision = str(final_rows[final_key]["decision"])
    return {
        "domain": spec.domain,
        "source_report": str(spec.source_report.relative_to(ROOT)),
        "raw_arm": spec.raw_arm,
        "curated_arm": spec.curated_arm,
        "raw": raw,
        "curated": curated,
        "curated_decision": curated_decision,
        "required_curated_decision": spec.required_curated_decision,
        "decision_matches_contract": curated_decision == spec.required_curated_decision,
        "record_reduction_fraction": _fraction_reduced(raw.get("records"), curated.get("records")),
        "token_proxy_reduction_fraction": _fraction_reduced(
            raw.get("token_proxy_count"), curated.get("token_proxy_count")
        ),
        "packed_token_reduction_fraction": _fraction_reduced(
            raw["packed_training_tokens"], curated["packed_training_tokens"]
        ),
    }


def _drift(raw_mix: JsonMap, curated_mix: JsonMap) -> JsonMap:
    domains = sorted(set(raw_mix["shares"]) | set(curated_mix["shares"]))
    return {
        domain: float(curated_mix["shares"].get(domain, 0.0)) - float(raw_mix["shares"].get(domain, 0.0))
        for domain in domains
    }


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Domain Composition Audit",
        "",
        f"Status: `{report['status']}`",
        f"Contract mode: `{report['contract_mode']}`",
        f"Target mix: `{report['target_domain_mix_status']}`",
        "",
        "| Domain | Raw tokens | Curated tokens | Token reduction | Raw share | Curated share | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    raw_share = report["mixes"]["raw"]["shares"]
    curated_share = report["mixes"]["curated"]["shares"]
    for row in report["domain_rows"]:
        domain = row["domain"]
        reduction = row["packed_token_reduction_fraction"]
        reduction_text = "" if reduction is None else f"{reduction:.6f}"
        lines.append(
            "| "
            f"{domain} | "
            f"{row['raw']['packed_training_tokens']} | "
            f"{row['curated']['packed_training_tokens']} | "
            f"{reduction_text} | "
            f"{raw_share[domain]:.6f} | "
            f"{curated_share[domain]:.6f} | "
            f"{row['curated_decision']} |"
        )
    lines.extend(["", "## Claim Boundary", "", report["claim_boundary"], ""])
    return "\n".join(lines)


def build() -> JsonMap:
    contract = load_json(CONTRACT_PATH)
    final_table = load_json(FINAL_TABLE_PATH)
    final_rows = _table_rows(final_table)
    domain_rows = [_domain_row(spec, final_rows) for spec in _domain_specs(contract)]
    raw_mix = _mix(domain_rows, "raw")
    curated_mix = _mix(domain_rows, "curated")
    target_declared = contract.get("target_domain_mix") is not None
    decision_contract_pass = all(row["decision_matches_contract"] for row in domain_rows)
    report = {
        "schema_version": "domain-composition-audit-v1",
        "status": (
            "domain_composition_audit_completed"
            if decision_contract_pass
            else "domain_composition_audit_blocked_decision_contract"
        ),
        "contract_mode": contract["contract_mode"],
        "canonical_claim": contract["canonical_claim"],
        "target_domain_mix_status": (
            "declared" if target_declared else "not_declared_for_current_paper_evidence"
        ),
        "domain_rows": domain_rows,
        "mixes": {
            "raw": raw_mix,
            "curated": curated_mix,
            "curated_minus_raw_share_drift": _drift(raw_mix, curated_mix),
        },
        "decision_contract_pass": decision_contract_pass,
        "forbidden_claims": contract["forbidden_claims"],
        "utility_scope": contract["utility_scope"],
        "source_sha256": {
            str(CONTRACT_PATH.relative_to(ROOT)): sha256_file(CONTRACT_PATH),
            str(FINAL_TABLE_PATH.relative_to(ROOT)): sha256_file(FINAL_TABLE_PATH),
            **{
                str(spec.source_report.relative_to(ROOT)): sha256_file(spec.source_report)
                for spec in _domain_specs(contract)
            },
        },
        "claim_boundary": contract["claim_boundary"],
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
