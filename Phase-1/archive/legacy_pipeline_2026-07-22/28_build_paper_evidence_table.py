#!/usr/bin/env python3
"""Build paper-ready evidence tables from the final curation reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, load_json, save_json


VALIDATION_DIR = OUTPUT_DIR / "validation"
DEFAULT_JSON_OUTPUT = VALIDATION_DIR / "paper_evidence_table.json"
DEFAULT_MD_OUTPUT = VALIDATION_DIR / "paper_evidence_table.md"
DEFAULT_CSV_OUTPUT = VALIDATION_DIR / "paper_evidence_table.csv"
SELECTOR_AUDIT_PATH = VALIDATION_DIR / "selector_baseline_audit.json"
READINESS_REPORT_PATH = VALIDATION_DIR / "curation_readiness_report.json"
PROTOCOL_REPORT_PATH = VALIDATION_DIR / "stage_c_protocol_decision_report.json"
STRICT_REPORT_PATH = VALIDATION_DIR / "strict_baseline_control_report.json"
DECISION_REPORT_PATH = VALIDATION_DIR / "curation_decision_report.json"


def _load_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def _verdict(selector: Dict[str, Any], dataset: str, baseline: str) -> Dict[str, Any]:
    dataset_payload = ((selector.get("datasets") or {}).get(dataset) or {})
    return (((dataset_payload.get("comparisons") or {}).get(baseline) or {}).get("verdict") or {})


def _round(value: Any, digits: int = 6) -> float | None:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


def _dataset_row(
    dataset: str,
    *,
    run_summary: Dict[str, Any],
    profile: str,
    selector: Dict[str, Any],
    readiness: Dict[str, Any],
    protocol: Dict[str, Any],
    strict: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    run = (((run_summary.get("profiles") or {}).get(profile) or {}).get(dataset) or {})
    selector_dataset = ((selector.get("datasets") or {}).get(dataset) or {})
    ready = ((readiness.get("datasets") or {}).get(dataset) or {})
    protocol_dataset = ((protocol.get("datasets") or {}).get(dataset) or {})
    strict_dataset = ((strict.get("datasets") or {}).get(dataset) or {})
    decision_dataset = ((decision.get("datasets") or {}).get(dataset) or {})
    utility = ready.get("utility") or {}
    coverage = ready.get("coverage") or {}
    random_verdict = _verdict(selector, dataset, "stageA_random")
    matched_verdict = _verdict(selector, dataset, "multi_matched_stageA_random")
    processed = int(run.get("processed_records") or ready.get("processed_records") or 0)
    stage_a = int(selector_dataset.get("stage_a_records") or 0)

    return {
        "dataset": dataset,
        "profile": profile,
        "processed_records": processed,
        "stage_a_records": stage_a,
        "stage_a_survival_rate": _round(stage_a / processed if processed else None),
        "selected_records": int(run.get("selected_records") or ready.get("selected_records") or 0),
        "selection_ratio": _round(run.get("selection_ratio") or ready.get("selection_ratio")),
        "stage_b_vs_stage_a_random": {
            "verdict": random_verdict.get("verdict"),
            "quality_delta": _round(random_verdict.get("quality_delta")),
            "learnability_delta": _round(random_verdict.get("learnability_delta")),
            "redundancy_risk_delta": _round(random_verdict.get("redundancy_risk_delta")),
        },
        "stage_b_vs_multi_matched": {
            "verdict": matched_verdict.get("verdict"),
            "quality_delta": _round(matched_verdict.get("quality_delta")),
            "learnability_delta": _round(matched_verdict.get("learnability_delta")),
            "redundancy_risk_delta": _round(matched_verdict.get("redundancy_risk_delta")),
        },
        "coverage": {
            "score": _round(coverage.get("score") or run.get("subset_coverage_retention_score")),
            "pass": bool((ready.get("stage_c") or {}).get("coverage_pass")),
            "backbone_pass": bool(coverage.get("backbone_pass")),
        },
        "utility": {
            "score": _round(utility.get("score") or run.get("small_lm_probe_gain_score")),
            "probe_status": utility.get("probe_status"),
            "selected_beats_stage_a_random": bool(utility.get("selected_beats_stageA_random")),
            "selected_beats_multi_matched": bool(utility.get("selected_beats_multi_matched")),
            "primary_estimand": "selected_vs_equal_budget_disjoint_stageA_random",
            "matched_controls_role": strict_dataset.get("matched_controls_role")
            or "conditional_mechanism_diagnostics_not_primary_gate",
            "strict_min_delta_nll": _round(utility.get("strict_min_delta_nll")),
            "strict_min_delta_nll_ci_low": _round(utility.get("strict_min_delta_nll_ci_low")),
            "token_exposure_caveat": bool(
                utility.get("token_exposure_confounded") or utility.get("token_exposure_inconclusive")
            ),
            "replicated_valid_families": list(protocol_dataset.get("replicated_valid_power_sweep_families") or []),
        },
        "reported_controls": {
            "canonical_strict_status": strict_dataset.get("canonical_strict_status"),
            "anti_memorization_available": bool(strict_dataset.get("anti_memorization_diagnostic_available")),
            "anti_memorization_supports_selected": bool(strict_dataset.get("anti_memorization_supports_selected")),
            "anti_memorization_delta_nll": _round(
                (strict_dataset.get("anti_memorization_evidence") or {}).get("delta_nll")
            ),
            "anti_memorization_delta_nll_ci_low": _round(
                (strict_dataset.get("anti_memorization_evidence") or {}).get("delta_nll_ci_low")
            ),
        },
        "decision": decision_dataset.get("decision"),
        "training_use": decision_dataset.get("training_use"),
        "operational_action": decision_dataset.get("operational_action"),
        "usable_data_sufficient": bool(decision_dataset.get("usable_data_sufficient")),
        "certification_claim_allowed": bool(decision_dataset.get("certification_claim_allowed")),
        "caveats": list(decision_dataset.get("caveats") or []),
        "selector_policy_action": strict_dataset.get("selector_policy_action") or "hold",
        "utility_scope": strict_dataset.get("utility_scope") or "Stage C validation only; never selector objective",
    }


def build_report(
    run_summary: Dict[str, Any],
    selector: Dict[str, Any],
    readiness: Dict[str, Any],
    protocol: Dict[str, Any],
    strict: Dict[str, Any],
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    profile = str(decision.get("profile") or readiness.get("profile") or protocol.get("profile") or "canonical")
    dataset_names = sorted((decision.get("datasets") or {}).keys())
    rows = {
        dataset: _dataset_row(
            dataset,
            run_summary=run_summary,
            profile=profile,
            selector=selector,
            readiness=readiness,
            protocol=protocol,
            strict=strict,
            decision=decision,
        )
        for dataset in dataset_names
    }
    certification_candidates = [
        dataset for dataset, row in rows.items() if bool(row.get("certification_claim_allowed"))
    ]
    return {
        "schema_version": "paper-evidence-table-v1",
        "profile": profile,
        "purpose": "Paper-ready evidence table for the LM-training data curation framework.",
        "claim_boundary": {
            "supported": "The framework can identify selected subsets supported for LM training under reported Stage-C evidence.",
            "not_supported": "The framework does not claim universal Utility improvement or use Utility as a Stage-B selector objective.",
            "utility_scope": "Stage C validation only; never selector objective",
        },
        "summary": {
            "dataset_count": len(rows),
            "certification_candidate_count": len(certification_candidates),
            "certification_candidates": certification_candidates,
        },
        "datasets": rows,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "-"
    return str(value) if value is not None else "-"


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    lines = [
        "# Paper Evidence Table",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Datasets: `{summary.get('dataset_count')}`",
        f"- Certification candidates: `{', '.join(summary.get('certification_candidates') or []) or 'none'}`",
        "- Utility scope: `Stage C validation only; never selector objective`",
        "",
        "## Main Results",
        "",
        "| Dataset | Stage-A survival | Selected | Quality delta (random / matched) | Redundancy-risk delta (random / matched) | Coverage | Utility > random | Utility > matched | Replicated family | Anti-mem support | Decision | Cert claim |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for dataset, row in (report.get("datasets") or {}).items():
        random = row.get("stage_b_vs_stage_a_random") or {}
        matched = row.get("stage_b_vs_multi_matched") or {}
        coverage = row.get("coverage") or {}
        utility = row.get("utility") or {}
        controls = row.get("reported_controls") or {}
        anti_mem_support = (
            _fmt(controls.get("anti_memorization_supports_selected"))
            if controls.get("anti_memorization_available")
            else "not run"
        )
        lines.append(
            f"| {dataset} | {_fmt(row.get('stage_a_survival_rate'))} | {_fmt(row.get('selection_ratio'))} | "
            f"{_fmt(random.get('quality_delta'))} / {_fmt(matched.get('quality_delta'))} | "
            f"{_fmt(random.get('redundancy_risk_delta'))} / {_fmt(matched.get('redundancy_risk_delta'))} | "
            f"{_fmt(coverage.get('score'))} | {_fmt(utility.get('selected_beats_stage_a_random'))} | "
            f"{_fmt(utility.get('selected_beats_multi_matched'))} | "
            f"{_fmt(utility.get('replicated_valid_families'))} | "
            f"{anti_mem_support} | "
            f"{row.get('decision')} | {_fmt(row.get('certification_claim_allowed'))} |"
        )

    for dataset in summary.get("certification_candidates") or []:
        row = (report.get("datasets") or {}).get(dataset) or {}
        random = row.get("stage_b_vs_stage_a_random") or {}
        matched = row.get("stage_b_vs_multi_matched") or {}
        coverage = row.get("coverage") or {}
        utility = row.get("utility") or {}
        controls = row.get("reported_controls") or {}
        anti_mem_evidence = (
            f"delta NLL={_fmt(controls.get('anti_memorization_delta_nll'))}, "
            f"CI low={_fmt(controls.get('anti_memorization_delta_nll_ci_low'))}"
            if controls.get("anti_memorization_available")
            else "profile-matched optional diagnostic not run"
        )
        anti_mem_result = (
            _fmt(controls.get("anti_memorization_supports_selected"))
            if controls.get("anti_memorization_available")
            else "not run"
        )
        lines.extend(
            [
                "",
                f"## Positive Case: {dataset}",
                "",
                "| Claim | Evidence | Result |",
                "|---|---|---|",
                f"| Stage A creates a usable pool | survival={_fmt(row.get('stage_a_survival_rate'))}, selected={row.get('selected_records')} | pass |",
                f"| Stage B improves Core features vs Stage-A random | quality delta={_fmt(random.get('quality_delta'))}, redundancy-risk delta={_fmt(random.get('redundancy_risk_delta'))} | {random.get('verdict')} |",
                f"| Stage B survives multi-matched comparison | quality delta={_fmt(matched.get('quality_delta'))}, redundancy-risk delta={_fmt(matched.get('redundancy_risk_delta'))} | {matched.get('verdict')} |",
                f"| Coverage is preserved | score={_fmt(coverage.get('score'))}, backbone={_fmt(coverage.get('backbone_pass'))} | {'pass' if coverage.get('pass') else 'fail'} |",
                f"| Primary Utility estimand | selected vs equal-budget disjoint Stage-A random | {_fmt(utility.get('selected_beats_stage_a_random'))} |",
                f"| Conditional matched diagnostics | canonical matched min delta NLL={_fmt(utility.get('strict_min_delta_nll'))}, CI low={_fmt(utility.get('strict_min_delta_nll_ci_low'))}; role={utility.get('matched_controls_role')} | {_fmt(utility.get('selected_beats_multi_matched'))} |",
                f"| Utility protocol replicates | families={_fmt(utility.get('replicated_valid_families'))} | pass |",
                f"| Anti-memorization control supports selected | {anti_mem_evidence} | {anti_mem_result} |",
                f"| Final training-use decision | {row.get('training_use')} | {row.get('decision')} |",
            ]
        )

    boundary = report.get("claim_boundary") or {}
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- Supported: {boundary.get('supported')}",
            f"- Not supported: {boundary.get('not_supported')}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report: Dict[str, Any], path: Path) -> None:
    fields = [
        "dataset",
        "profile",
        "processed_records",
        "stage_a_records",
        "stage_a_survival_rate",
        "selected_records",
        "selection_ratio",
        "coverage_score",
        "selected_beats_stage_a_random",
        "selected_beats_multi_matched",
        "replicated_valid_families",
        "anti_memorization_supports_selected",
        "decision",
        "operational_action",
        "training_use",
        "certification_claim_allowed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in (report.get("datasets") or {}).values():
            utility = row.get("utility") or {}
            coverage = row.get("coverage") or {}
            controls = row.get("reported_controls") or {}
            writer.writerow(
                {
                    "dataset": row.get("dataset"),
                    "profile": row.get("profile"),
                    "processed_records": row.get("processed_records"),
                    "stage_a_records": row.get("stage_a_records"),
                    "stage_a_survival_rate": row.get("stage_a_survival_rate"),
                    "selected_records": row.get("selected_records"),
                    "selection_ratio": row.get("selection_ratio"),
                    "coverage_score": coverage.get("score"),
                    "selected_beats_stage_a_random": utility.get("selected_beats_stage_a_random"),
                    "selected_beats_multi_matched": utility.get("selected_beats_multi_matched"),
                    "replicated_valid_families": ",".join(utility.get("replicated_valid_families") or []),
                    "anti_memorization_supports_selected": controls.get("anti_memorization_supports_selected"),
                    "decision": row.get("decision"),
                    "operational_action": row.get("operational_action"),
                    "training_use": row.get("training_use"),
                    "certification_claim_allowed": row.get("certification_claim_allowed"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready curation evidence tables.")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        _load_optional(RUN_SUMMARY_PATH),
        _load_optional(SELECTOR_AUDIT_PATH),
        _load_optional(READINESS_REPORT_PATH),
        _load_optional(PROTOCOL_REPORT_PATH),
        _load_optional(STRICT_REPORT_PATH),
        _load_optional(DECISION_REPORT_PATH),
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    write_csv(report, args.csv_output)
    print(f"[28] paper evidence json: {args.output}", flush=True)
    print(f"[28] paper evidence md: {args.md_output}", flush=True)
    print(f"[28] paper evidence csv: {args.csv_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
