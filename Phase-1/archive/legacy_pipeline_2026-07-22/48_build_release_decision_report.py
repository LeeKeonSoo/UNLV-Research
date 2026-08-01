#!/usr/bin/env python3
"""Build a deployment-contract-conditioned release decision report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json
from release_policy import decide_release


DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "release_decision_report.json"


def build_report(contract: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    decision = decide_release(contract, evidence)
    return {
        "schema_version": "release-decision-report-v1",
        "deployment_contract": contract,
        "evidence_identity": evidence.get("evidence_identity"),
        "decision": decision,
        "framework_contract": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selected core; no Utility objective",
            "stage_c": "subset-level validation",
            "release_layer": "deployment-contract-conditioned release selection or abstention",
            "utility_scope": "Stage C validation only; never selector objective",
        },
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    contract = report["deployment_contract"]
    decision = report["decision"]
    lines = [
        "# Release Decision Report",
        "",
        f"- Contract: `{contract.get('contract_name')}`",
        f"- Objective: `{contract.get('objective_type')}`",
        f"- Release action: `{decision.get('release_action')}`",
        f"- Supported: `{decision.get('supported')}`",
        f"- Claim scope: {decision.get('claim_scope')}",
        "",
        "## Rationale",
        "",
        str(decision.get("rationale") or ""),
        "",
        "## Candidate Assessments",
        "",
        "| Release | Eligible | Primary improvement | Reasons |",
        "| --- | --- | ---: | --- |",
    ]
    for action, payload in (decision.get("candidate_assessments") or {}).items():
        gain = payload.get("primary_improvement")
        gain_cell = f"{gain:.9f}" if isinstance(gain, float) else "missing"
        lines.append(f"| `{action}` | {payload.get('eligible')} | {gain_cell} | {', '.join(payload.get('reasons') or []) or 'none'} |")
    lines.extend(["", "## Contract Boundary", "", str(contract.get("claim_scope") or ""), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deployment-conditioned release decision.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path)
    args = parser.parse_args()
    report = build_report(load_json(args.contract), load_json(args.evidence))
    save_json(args.output, report)
    md_output = args.md_output or args.output.with_suffix(".md")
    write_markdown(report, md_output)
    print({"release_action": report["decision"]["release_action"], "supported": report["decision"]["supported"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
