#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

DEFAULT_V2_DECISION = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_GAP = OUTPUT_DIR / "validation" / "stage_c_guardrail_gap_report.json"
DEFAULT_TRAINING = OUTPUT_DIR / "validation" / "stage_c_training_validation_report.json"
DEFAULT_RELEASE_GATE = OUTPUT_DIR / "validation" / "paper_claim_release_gate_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "confirmatory_decision_boundary_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "confirmatory_decision_boundary_report.md"


def _as_map(value: JsonValue) -> JsonMap:
    return value if isinstance(value, dict) else {}


def _as_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _string_items(value: JsonValue) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _source(path: Path) -> JsonMap:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _load(path: Path) -> JsonMap:
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def _nll_evidence(v2_decision: JsonMap, training: JsonMap) -> JsonMap:
    summary = _as_map(v2_decision.get("summary"))
    nll_gate = _as_map(summary.get("nll_gate"))
    training_v2 = _as_map(training.get("v2_confirmatory_training"))
    return {
        "decision_status": v2_decision.get("status"),
        "training_runs_completed": summary.get("training_runs_completed"),
        "expected_training_runs": summary.get("expected_training_runs"),
        "heldout_nll_results_completed": summary.get("heldout_nll_results_completed"),
        "expected_heldout_nll_results": summary.get("expected_heldout_nll_results"),
        "nll_gate_status": nll_gate.get("status") or training_v2.get("nll_gate_status"),
        "curated_vs_stageA_random_mean_nll_reduction": nll_gate.get("curated_vs_stageA_random_mean_nll_reduction")
        or training_v2.get("curated_vs_stageA_random_mean_nll_reduction"),
        "curated_vs_raw_random_mean_nll_reduction": nll_gate.get("curated_vs_raw_random_mean_nll_reduction")
        or training_v2.get("curated_vs_raw_random_mean_nll_reduction"),
        "curated_vs_stageA_random_margin_pass": nll_gate.get("curated_vs_stageA_random_margin_pass"),
        "curated_vs_stageA_random_all_paired_seed_pass": nll_gate.get("curated_vs_stageA_random_all_paired_seed_pass"),
    }


def _guardrail_decision(gap: JsonMap) -> JsonMap:
    guardrails = _as_map(gap.get("guardrails"))
    incomplete = _string_items(gap.get("incomplete_guardrails"))
    failed = _string_items(gap.get("failed_guardrails"))
    passed = [
        name
        for name, raw_row in guardrails.items()
        if str(_as_map(raw_row).get("status") or "").endswith("_passed")
    ]
    return {
        "gap_status": gap.get("status"),
        "incomplete_guardrails": incomplete,
        "failed_guardrails": failed,
        "passed_guardrails": passed,
        "missing_guardrail_action": "abstain",
    }


def _boundary_status(missing: list[str], required_guardrails_complete: bool, release_gate: JsonMap) -> str:
    if missing:
        return "confirmatory_decision_blocked_missing_inputs"
    if not required_guardrails_complete:
        return "confirmatory_decision_abstain_required_guardrails_incomplete"
    if release_gate.get("status") != "paper_curation_stage_claim_gate_passed":
        return "confirmatory_decision_abstain_release_gate_blocked"
    return "confirmatory_decision_curation_stage_claim_passed"


def _final_decision(status: str) -> str:
    if status == "confirmatory_decision_curation_stage_claim_passed":
        return "curation_stage_claim_pass"
    return "abstain_not_release_pass"


def _allowed_claims(required_guardrails_complete: bool, release_supported: bool) -> list[str]:
    claims = [
        "frozen_v2_confirmatory_target_code_nll_gate_passed",
        "general_text_confirmatory_guardrail_passed",
    ]
    if required_guardrails_complete:
        claims.extend(
            [
                "required_stage_c_confirmatory_guardrails_complete",
                "confirmatory_guardrails_complete_release_still_blocked_by_claim_gate",
            ]
        )
    else:
        claims.append("confirmatory_decision_must_abstain_until_required_guardrails_complete")
    if release_supported:
        claims.append("curation_stage_paper_claim_pass")
    return claims


def _remaining_actions(required_guardrails_complete: bool, release_gate: JsonMap) -> list[str]:
    if not required_guardrails_complete:
        return [
            "complete_evalplus_confirmatory_guardrail_for_all_required_arms_and_seeds",
            "complete_general_task_confirmatory_retention_for_all_required_arms_and_seeds",
            "rebuild_v2_confirmatory_decision_report_after_guardrails_complete",
            "rerun_paper_claim_release_gate",
        ]
    blockers = set(_string_items(release_gate.get("blockers")))
    actions: list[str] = []
    production_blockers = set(_string_items(release_gate.get("production_blockers")))
    if "production_core_validity_not_supported" in production_blockers:
        actions.append("complete_production_core_validity_before_deployment_claims")
    if any(item.startswith("canonical_guardrail_release_not_supported") for item in blockers):
        actions.append("complete_canonical_guardrail_release_evidence")
    if any(item.startswith("target_size_") for item in blockers):
        actions.append("complete_target_size_release_guardrails")
    if release_gate.get("status") != "paper_curation_stage_claim_gate_passed":
        actions.append("rerun_paper_claim_release_gate_after_release_blockers")
    return actions


def build(
    v2_decision_path: Path,
    gap_path: Path,
    training_path: Path,
    release_gate_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> JsonMap:
    v2_decision = _load(v2_decision_path)
    gap = _load(gap_path)
    training = _load(training_path)
    release_gate = _load(release_gate_path)
    missing = [
        name
        for name, path in {
            "v2_confirmatory_decision": v2_decision_path,
            "stage_c_guardrail_gap": gap_path,
            "stage_c_training_validation": training_path,
            "paper_claim_release_gate": release_gate_path,
        }.items()
        if not path.exists()
    ]
    nll = _nll_evidence(v2_decision, training)
    guardrails = _guardrail_decision(gap)
    required_guardrails_complete = not guardrails["incomplete_guardrails"] and not guardrails["failed_guardrails"]
    nll_supported = (
        nll.get("nll_gate_status") == "passed"
        and bool(nll.get("curated_vs_stageA_random_margin_pass"))
        and bool(nll.get("curated_vs_stageA_random_all_paired_seed_pass"))
    )
    release_supported = release_gate.get("status") == "paper_curation_stage_claim_gate_passed"
    status = _boundary_status(missing, bool(required_guardrails_complete), release_gate)
    report = {
        "schema_version": "confirmatory-decision-boundary-report-v1",
        "status": status,
        "final_decision": _final_decision(status),
        "sources": {
            "v2_confirmatory_decision": _source(v2_decision_path),
            "stage_c_guardrail_gap": _source(gap_path),
            "stage_c_training_validation": _source(training_path),
            "paper_claim_release_gate": _source(release_gate_path),
        },
        "missing_inputs": missing,
        "nll_evidence": nll,
        "guardrail_decision": guardrails,
        "claim_decision": {
            "target_nll_confirmatory_effect_supported": bool(nll_supported),
            "required_confirmatory_guardrails_complete": bool(required_guardrails_complete),
            "curation_stage_claim_supported": bool(release_supported),
            "production_deployment_claim_supported": bool(release_gate.get("production_deployment_claim_supported")),
            "stage_b_tuning_allowed": False,
            "utility_in_selector_supported": False,
        },
        "release_gate_status": release_gate.get("status"),
        "release_gate_blockers": release_gate.get("blockers"),
        "production_blockers": release_gate.get("production_blockers"),
        "allowed_claims": _allowed_claims(bool(required_guardrails_complete), bool(release_supported)),
        "forbidden_claims": [
            "production_ready_framework",
            "complete_confirmatory_validation",
            "tuning_stage_b_from_confirmatory_outcomes",
            "treating_incomplete_guardrails_as_failed_or_passed",
        ],
        "remaining_actions": _remaining_actions(bool(required_guardrails_complete), release_gate),
        "utility_scope": "Stage C validation only; never selector objective.",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Confirmatory Decision Boundary Report",
        "",
        f"Status: `{report.get('status')}`",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        "## Claim Decision",
        "",
    ]
    for key, value in _as_map(report.get("claim_decision")).items():
        lines.append(f"- `{key}`: `{value}`")
    guardrails = _as_map(report.get("guardrail_decision"))
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            f"- Passed: `{', '.join(_string_items(guardrails.get('passed_guardrails'))) or 'None'}`",
            f"- Incomplete: `{', '.join(_string_items(guardrails.get('incomplete_guardrails'))) or 'None'}`",
            f"- Failed: `{', '.join(_string_items(guardrails.get('failed_guardrails'))) or 'None'}`",
            "",
            "## Remaining Actions",
            "",
        ]
    )
    lines.extend([f"- `{item}`" for item in _string_items(report.get("remaining_actions"))])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in _string_items(report.get("forbidden_claims"))])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the confirmatory decision boundary report.")
    parser.add_argument("--v2-decision", type=Path, default=DEFAULT_V2_DECISION)
    parser.add_argument("--gap", type=Path, default=DEFAULT_GAP)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--release-gate", type=Path, default=DEFAULT_RELEASE_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.v2_decision, args.gap, args.training, args.release_gate, args.output, args.md_output)
    print({"status": report.get("status"), "final_decision": report.get("final_decision")})
    return 0 if not _as_list(report.get("missing_inputs")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
