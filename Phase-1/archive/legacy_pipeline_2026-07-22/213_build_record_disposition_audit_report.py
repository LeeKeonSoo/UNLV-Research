#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, PROJECT_DIR, load_json, save_json
from policy.dispositions import (
    BUDGET_NOT_REQUESTED,
    BUDGET_NOT_SELECTED,
    BUDGET_SELECTED,
    CURATION_QUARANTINED,
    CURATION_REJECTED,
    CURATION_RETAINED,
    annotate_retained_pool,
    disposition_summary,
)


JsonMap = dict[str, Any]

DEFAULT_CONTRACT = PROJECT_DIR / "configs" / "lm_curation_operational_framework_v1.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "record_disposition_audit_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "record_disposition_audit_report.md"


def _base_records() -> list[JsonMap]:
    return [
        {"chunk_uid": "selected-a", "text": "usable selected"},
        {"chunk_uid": "selected-b", "text": "usable selected"},
        {"chunk_uid": "not-selected", "text": "usable retained outside budget"},
    ]


def _non_retained_examples() -> list[JsonMap]:
    return [
        {
            "chunk_uid": "hard-rejected",
            "curation_decision": {
                "curation_disposition": CURATION_REJECTED,
                "curation_reason": "failed_stage_a_hard_gate",
                "training_budget_disposition": BUDGET_NOT_REQUESTED,
                "budget_exclusion_is_rejection": False,
            },
        },
        {
            "chunk_uid": "quarantined",
            "curation_decision": {
                "curation_disposition": CURATION_QUARANTINED,
                "curation_reason": "stage0_risk_quarantine",
                "training_budget_disposition": BUDGET_NOT_REQUESTED,
                "budget_exclusion_is_rejection": False,
            },
        },
    ]


def _allowed_sets(contract: JsonMap) -> tuple[set[str], set[str], set[str]]:
    dispositions = contract["disposition_contract"]
    actions = contract["stage_contract"]["decision_release"]["allowed_actions"]
    return (
        set(dispositions["curation_dispositions"]),
        set(dispositions["training_budget_dispositions"]),
        set(actions),
    )


def build(output_path: Path, md_output_path: Path) -> JsonMap:
    contract = load_json(DEFAULT_CONTRACT)
    allowed_curation, allowed_budget, allowed_actions = _allowed_sets(contract)
    selected_ids = {"selected-a", "selected-b"}
    budgeted = annotate_retained_pool(_base_records(), selected_ids=selected_ids, budget_applied=True)
    retain_all = annotate_retained_pool(_base_records(), selected_ids={row["chunk_uid"] for row in _base_records()}, budget_applied=False)
    all_examples = [*budgeted, *retain_all, *_non_retained_examples()]
    summary = disposition_summary(all_examples)
    observed_curation = set(summary["curation_disposition_counts"])
    observed_budget = set(summary["training_budget_disposition_counts"])
    budget_not_selected_rows = [
        row
        for row in budgeted
        if row["curation_decision"]["training_budget_disposition"] == BUDGET_NOT_SELECTED
    ]
    retain_all_is_valid = all(
        row["curation_decision"]["curation_disposition"] == CURATION_RETAINED
        and row["curation_decision"]["training_budget_disposition"] == BUDGET_NOT_REQUESTED
        for row in retain_all
    )
    budget_not_selected_is_rejection = any(
        row["curation_decision"]["curation_disposition"] == CURATION_REJECTED
        or row["curation_decision"]["budget_exclusion_is_rejection"] is True
        for row in budget_not_selected_rows
    )
    blockers = [
        name
        for name, passed in {
            "curation_dispositions_not_allowed": observed_curation <= allowed_curation,
            "budget_dispositions_not_allowed": observed_budget <= allowed_budget,
            "missing_budget_not_selected_fixture": bool(budget_not_selected_rows),
            "budget_not_selected_treated_as_rejection": not budget_not_selected_is_rejection,
            "retain_all_semantics_failed": retain_all_is_valid,
            "abstain_action_missing": "abstain" in allowed_actions,
            "retain_all_action_missing": "retain_all" in allowed_actions,
        }.items()
        if not passed
    ]
    report = {
        "schema_version": "record-disposition-audit-report-v1",
        "status": "record_disposition_audit_passed" if not blockers else "record_disposition_audit_failed",
        "budget_not_selected_is_rejection": budget_not_selected_is_rejection,
        "retain_all_is_valid": retain_all_is_valid,
        "abstain_action_allowed": "abstain" in allowed_actions,
        "observed_curation_dispositions": sorted(observed_curation),
        "observed_training_budget_dispositions": sorted(observed_budget),
        "summary": summary,
        "blockers": blockers,
        "claim_boundary": (
            "Record-level disposition audit only. This verifies decision semantics, "
            "not Stage-C Utility or production deployment readiness."
        ),
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Record Disposition Audit Report",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Semantics",
        "",
        f"- `budget_not_selected_is_rejection`: `{report['budget_not_selected_is_rejection']}`",
        f"- `retain_all_is_valid`: `{report['retain_all_is_valid']}`",
        f"- `abstain_action_allowed`: `{report['abstain_action_allowed']}`",
        "",
        "## Observed Dispositions",
        "",
        f"- Curation: `{', '.join(report['observed_curation_dispositions'])}`",
        f"- Training budget: `{', '.join(report['observed_training_budget_dispositions'])}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{item}`" for item in report["blockers"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build record-level disposition audit report.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if report["status"] == "record_disposition_audit_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
