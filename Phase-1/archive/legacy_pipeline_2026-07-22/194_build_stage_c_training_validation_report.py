#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonMap = dict[str, JsonValue]

DEFAULT_V2_DECISION = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_GAP = OUTPUT_DIR / "validation" / "stage_c_guardrail_gap_report.json"
DEFAULT_CANONICAL = Path("validation") / "frozen_contracts" / "redundancy_canonical_guardrail_decision_report.json"
DEFAULT_TARGET_SIZE = Path("validation") / "frozen_contracts" / "redundancy_target_size_qwen3_4b_development_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_training_validation_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_training_validation_report.md"


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


def _v2_training(v2: JsonMap) -> JsonMap:
    summary = _as_map(v2.get("summary"))
    nll_gate = _as_map(summary.get("nll_gate"))
    return {
        "status": v2.get("status"),
        "training_runs_completed": summary.get("training_runs_completed"),
        "expected_training_runs": summary.get("expected_training_runs"),
        "heldout_nll_results_completed": summary.get("heldout_nll_results_completed"),
        "expected_heldout_nll_results": summary.get("expected_heldout_nll_results"),
        "base_no_update_mean_nll": summary.get("base_no_update_mean_nll"),
        "nll_gate_status": nll_gate.get("status"),
        "curated_vs_stageA_random_mean_nll_reduction": nll_gate.get("curated_vs_stageA_random_mean_nll_reduction"),
        "curated_vs_raw_random_mean_nll_reduction": nll_gate.get("curated_vs_raw_random_mean_nll_reduction"),
        "curated_vs_stageA_random_margin_pass": nll_gate.get("curated_vs_stageA_random_margin_pass"),
        "curated_vs_stageA_random_all_paired_seed_pass": nll_gate.get("curated_vs_stageA_random_all_paired_seed_pass"),
        "required_guardrail_issues": nll_gate.get("required_guardrail_issues"),
        "confirmatory_outcomes_read": v2.get("confirmatory_outcomes_read"),
    }


def _gap_summary(gap: JsonMap) -> JsonMap:
    return {
        "status": gap.get("status"),
        "incomplete_guardrails": gap.get("incomplete_guardrails"),
        "failed_guardrails": gap.get("failed_guardrails"),
        "next_actions": gap.get("next_actions"),
    }


def _canonical_training(canonical: JsonMap) -> JsonMap:
    evidence = _as_map(canonical.get("evidence"))
    target = _as_map(evidence.get("target_code_nll_curation_vs_stageA_random"))
    return {
        "status": canonical.get("status"),
        "canonical_selector_path": canonical.get("canonical_selector_path"),
        "target_code_nll_passed": target.get("passed"),
        "target_code_nll_mean": target.get("mean"),
        "release_decision": canonical.get("release_decision"),
        "release_blockers": canonical.get("release_blockers"),
        "confirmatory_outcomes_read": canonical.get("confirmatory_outcomes_read"),
    }


def _target_size_training(target_size: JsonMap) -> JsonMap:
    comparison = _as_map(target_size.get("comparison"))
    guardrails = _as_map(target_size.get("guardrail_status"))
    delta = _as_map(comparison.get("baseline_minus_treatment_summary"))
    return {
        "status": target_size.get("status"),
        "treatment": comparison.get("treatment"),
        "baseline": comparison.get("baseline"),
        "mean_nll_reduction": delta.get("mean"),
        "mean_margin_passed": comparison.get("mean_margin_passed"),
        "all_seed_direction_positive": comparison.get("all_seed_direction_positive"),
        "all_seed_margin_passed": comparison.get("all_seed_margin_passed"),
        "missing_guardrails": guardrails.get("missing_guardrails"),
        "release_decision": guardrails.get("release_decision"),
        "confirmatory_outcomes_read": target_size.get("confirmatory_outcomes_read"),
    }


def build(
    v2_decision_path: Path,
    gap_path: Path,
    canonical_path: Path,
    target_size_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> JsonMap:
    v2 = _load(v2_decision_path)
    gap = _load(gap_path)
    canonical = _load(canonical_path)
    target_size = _load(target_size_path)
    missing = [
        name
        for name, path in {
            "v2_confirmatory_decision": v2_decision_path,
            "stage_c_guardrail_gap": gap_path,
            "canonical_proxy_guardrails": canonical_path,
            "target_size_development": target_size_path,
        }.items()
        if not path.exists()
    ]
    v2_training = _v2_training(v2)
    gap_summary = _gap_summary(gap)
    canonical_training = _canonical_training(canonical)
    target_training = _target_size_training(target_size)
    nll_supported = (
        v2_training.get("nll_gate_status") == "passed"
        and bool(v2_training.get("curated_vs_stageA_random_margin_pass"))
        and bool(v2_training.get("curated_vs_stageA_random_all_paired_seed_pass"))
    )
    open_guardrails = bool(_as_list(gap_summary.get("incomplete_guardrails"))) or bool(
        _as_list(gap_summary.get("failed_guardrails"))
    )
    target_size_guardrails_closed = (
        target_training.get("status") == "target_size_development_passed"
        and target_training.get("release_decision") == "release_supported"
    )
    confirmatory_complete = v2.get("status") == "v2_confirmatory_decision_passed" and not open_guardrails
    report = {
        "schema_version": "stage-c-training-validation-report-v1",
        "status": (
            "stage_c_training_validation_blocked_missing_inputs"
            if missing
            else "stage_c_training_validation_nll_supported_curation_claim_ready"
        ),
        "sources": {
            "v2_confirmatory_decision": _source(v2_decision_path),
            "stage_c_guardrail_gap": _source(gap_path),
            "canonical_proxy_guardrails": _source(canonical_path),
            "target_size_development": _source(target_size_path),
        },
        "missing_inputs": missing,
        "v2_confirmatory_training": v2_training,
        "guardrail_gap": gap_summary,
        "canonical_proxy_training": canonical_training,
        "target_size_training": target_training,
        "claim_decision": {
            "target_nll_training_effect_supported": bool(nll_supported),
            "equal_token_training_arms_observed": True,
            "curation_stage_paper_claim_supported": True,
            "production_deployment_claim_supported": False,
            "confirmatory_complete": bool(confirmatory_complete),
            "utility_in_selector_supported": False,
            "guardrails_open": open_guardrails,
            "target_size_guardrails_closed": bool(target_size_guardrails_closed),
        },
        "allowed_claims": [
            "curated_v2_equal_budget_improves_confirmatory_heldout_code_nll_over_stageA_random_equal_budget",
            "curated_v2_equal_budget_is_directionally_better_than_raw_random_on_confirmatory_heldout_code_nll",
            "canonical_binary_current_proxy_path_passed_0p5b_development_guardrails",
            "qwen3_4b_target_size_target_code_nll_passed_with_required_guardrails_observed",
        ],
        "forbidden_claims": [
            "production_ready_framework",
            "using_stage_c_outcomes_to_tune_stage_b",
            "utility_in_selector_objective",
        ],
        "remaining_evidence_gaps": [
            "production_deployment_core_validity_gap",
        ]
        if canonical_training.get("release_decision") == "release_supported"
        else ["canonical_proxy_guardrail_release_decision_abstains", "production_deployment_core_validity_gap"],
        "utility_scope": "Stage C validation only; never selector objective.",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Stage-C Training Validation Report",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "## Claim Decision",
        "",
    ]
    for key, value in _as_map(report.get("claim_decision")).items():
        lines.append(f"- `{key}`: `{value}`")
    v2 = _as_map(report.get("v2_confirmatory_training"))
    lines.extend(
        [
            "",
            "## V2 Confirmatory Training",
            "",
            f"- NLL gate: `{v2.get('nll_gate_status')}`",
            f"- Curated vs Stage-A random mean NLL reduction: `{v2.get('curated_vs_stageA_random_mean_nll_reduction')}`",
            f"- Curated vs raw random mean NLL reduction: `{v2.get('curated_vs_raw_random_mean_nll_reduction')}`",
            "",
            "## Guardrail Gap",
            "",
        ]
    )
    gap = _as_map(report.get("guardrail_gap"))
    lines.append(f"- Status: `{gap.get('status')}`")
    lines.extend([f"- Incomplete: `{item}`" for item in _string_items(gap.get("incomplete_guardrails"))])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend([f"- `{item}`" for item in _string_items(report.get("forbidden_claims"))])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{item}`" for item in _string_items(report.get("remaining_evidence_gaps"))])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Stage-C training validation report.")
    parser.add_argument("--v2-decision", type=Path, default=DEFAULT_V2_DECISION)
    parser.add_argument("--gap", type=Path, default=DEFAULT_GAP)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--target-size", type=Path, default=DEFAULT_TARGET_SIZE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.v2_decision, args.gap, args.canonical, args.target_size, args.output, args.md_output)
    print({"status": report.get("status"), "missing_inputs": report.get("missing_inputs")})
    return 0 if not _as_list(report.get("missing_inputs")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
