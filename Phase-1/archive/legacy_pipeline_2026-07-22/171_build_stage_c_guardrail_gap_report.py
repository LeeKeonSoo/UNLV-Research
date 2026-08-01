#!/usr/bin/env python3
"""Summarize remaining Stage-C confirmatory guardrail work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_DECISION = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_EVALPLUS = OUTPUT_DIR / "validation" / "code_domain_v2_evalplus_confirmatory_guardrail_report.json"
DEFAULT_GENERAL_TASK = OUTPUT_DIR / "validation" / "code_domain_v2_general_task_confirmatory_guardrail_report.json"
DEFAULT_GENERAL_TEXT = OUTPUT_DIR / "validation" / "code_domain_v2_general_text_confirmatory_guardrail_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_guardrail_gap_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_guardrail_gap_report.md"


def _read(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def _blockers(report: Dict[str, Any] | None) -> List[str]:
    if not isinstance(report, dict):
        return ["report_missing"]
    return [str(item) for item in report.get("blockers") or []]


def _status(report: Dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return "missing"
    return str(report.get("status") or "unknown")


def build(
    decision_path: Path,
    evalplus_path: Path,
    general_task_path: Path,
    general_text_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    decision = _read(decision_path)
    evalplus = _read(evalplus_path)
    general_task = _read(general_task_path)
    general_text = _read(general_text_path)
    guardrails = {
        "evalplus_confirmatory": {
            "path": str(evalplus_path),
            "status": _status(evalplus),
            "blockers": _blockers(evalplus),
        },
        "general_task_retention": {
            "path": str(general_task_path),
            "status": _status(general_task),
            "blockers": _blockers(general_task),
        },
        "general_text_nll_retention": {
            "path": str(general_text_path),
            "status": _status(general_text),
            "blockers": _blockers(general_text),
        },
    }
    incomplete = [
        name
        for name, row in guardrails.items()
        if row["status"] == "missing" or row["status"].endswith("_incomplete")
    ]
    failed = [
        name
        for name, row in guardrails.items()
        if row["status"] not in {"missing"}
        and not row["status"].endswith("_incomplete")
        and not row["status"].endswith("_passed")
        and "passed" not in row["status"]
    ]
    next_actions = []
    if guardrails["evalplus_confirmatory"]["status"].endswith("_incomplete"):
        next_actions.append(
            "Complete EvalPlus confirmatory sample generation for every required arm/seed, then run Docker EvalPlus evaluation."
        )
    if guardrails["general_task_retention"]["status"].endswith("_incomplete"):
        next_actions.append(
            "Run remaining general-task lm-eval confirmatory jobs on CUDA device 1 and rebuild the guardrail report."
        )
    if failed:
        next_actions.append("Do not release; at least one complete guardrail failed.")
    if not next_actions:
        next_actions.append("Rebuild the v2 confirmatory decision report and inspect the final release status.")

    report = {
        "schema_version": "stage-c-guardrail-gap-report-v1",
        "status": "stage_c_guardrail_gaps_open" if incomplete else "stage_c_guardrail_gaps_closed",
        "decision_report": {
            "path": str(decision_path),
            "status": _status(decision),
            "nll_gate_status": ((decision or {}).get("summary") or {}).get("nll_gate", {}).get("status")
            if isinstance(decision, dict)
            else None,
        },
        "guardrails": guardrails,
        "incomplete_guardrails": incomplete,
        "failed_guardrails": failed,
        "next_actions": next_actions,
        "claim_boundary": "Stage-C work queue only; does not change Stage-B selection or release status.",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Stage-C Guardrail Gap Report",
        "",
        f"Status: `{report['status']}`",
        "",
        f"Decision status: `{report['decision_report']['status']}`",
        f"NLL gate: `{report['decision_report']['nll_gate_status']}`",
        "",
        "## Guardrails",
        "",
        "| Guardrail | Status | Blockers |",
        "| --- | --- | --- |",
    ]
    for name, row in report["guardrails"].items():
        lines.append(f"| `{name}` | `{row['status']}` | `{len(row['blockers'])}` |")
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {item}" for item in report["next_actions"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-C guardrail gap report.")
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--evalplus", type=Path, default=DEFAULT_EVALPLUS)
    parser.add_argument("--general-task", type=Path, default=DEFAULT_GENERAL_TASK)
    parser.add_argument("--general-text", type=Path, default=DEFAULT_GENERAL_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.decision, args.evalplus, args.general_task, args.general_text, args.output, args.md_output)
    print({"status": report["status"], "incomplete_guardrails": report["incomplete_guardrails"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
