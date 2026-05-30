#!/usr/bin/env python3
"""Compare a candidate run_summary against a saved canonical snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH

DEFAULT_BASELINE = OUTPUT_DIR / "validation" / "canonical_run_summary_snapshot_before_candidate.json"
DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "candidate_profile_comparison.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "candidate_profile_comparison.md"


def _profiles(summary: Dict[str, Any]) -> Dict[str, Any]:
    return summary.get("profiles") or {}


def _resolve_profile(summary: Dict[str, Any], requested: str) -> str:
    profiles = _profiles(summary)
    if requested in profiles:
        return requested
    names = [str(k) for k, v in profiles.items() if not str(k).startswith("_") and isinstance(v, dict)]
    if len(names) == 1:
        return names[0]
    raise RuntimeError(f"Profile {requested!r} not found. Available: {names}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _utility_aggregate(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (((meta.get("utility_probe_details") or {}).get("aggregate") or {}))


def _evidence(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (_utility_aggregate(meta).get("utility_evidence_summary") or {})


def _probe_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (_utility_aggregate(meta).get("probe_sensitivity_status") or {})


def _curation_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (_utility_aggregate(meta).get("curation_benefit_status") or {})


def _strict_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (_utility_aggregate(meta).get("strict_counterfactual_status") or {})


def _dataset_row(meta: Dict[str, Any]) -> Dict[str, Any]:
    stage_c = meta.get("stage_c_core_validation") or {}
    utility = _utility_aggregate(meta)
    evidence = _evidence(meta)
    strict_status = _strict_status(meta)
    return {
        "selected_records": int(meta.get("selected_records") or 0),
        "selection_ratio": _safe_float(meta.get("selection_ratio")),
        "coverage": _safe_float(meta.get("subset_coverage_retention_score")),
        "utility": _safe_float(meta.get("small_lm_probe_gain_score")),
        "strict_min_gain": _safe_float(evidence.get("strict_min_gain")),
        "stage_c_pass": bool(stage_c.get("passed")),
        "coverage_pass": bool((stage_c.get("coverage") or {}).get("passed")),
        "utility_pass": bool((stage_c.get("utility") or {}).get("passed")),
        "utility_probe_valid": utility.get("utility_probe_valid"),
        "utility_strict_pass": utility.get("utility_strict_pass"),
        "evidence_tier": utility.get("evidence_tier") or evidence.get("evidence_tier"),
        "failure_reason": (
            utility.get("failure_reason")
            or utility.get("utility_failure_reason")
            or evidence.get("failure_reason")
            or evidence.get("utility_failure_reason")
        ),
        "probe_valid": _probe_status(meta).get("probe_valid"),
        "selected_beats_random": _curation_status(meta).get("selected_beats_random"),
        "selected_beats_matched": strict_status.get("selected_beats_multi_matched"),
        "strict_status": strict_status.get("status"),
        "strict_min_delta_nll": _safe_float(evidence.get("strict_min_delta_nll")),
        "strict_min_delta_nll_ci_low": _safe_float(evidence.get("strict_min_delta_nll_ci_low")),
        "max_mde_95": _safe_float(evidence.get("max_minimum_detectable_delta_nll_95")),
        "min_effect_to_mde_ratio": _safe_float(evidence.get("min_effect_to_mde_ratio")),
        "noise_dominated": evidence.get("noise_dominated"),
    }


def _delta(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    keys = ["selected_records", "selection_ratio", "coverage", "utility", "strict_min_gain"]
    return {key: round(_safe_float(candidate.get(key)) - _safe_float(baseline.get(key)), 6) for key in keys}


def build_comparison(*, baseline_path: Path, current_path: Path, baseline_profile: str, candidate_profile: str) -> Dict[str, Any]:
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    current_summary = json.loads(current_path.read_text(encoding="utf-8"))
    baseline_profile = _resolve_profile(baseline_summary, baseline_profile)
    candidate_profile = _resolve_profile(current_summary, candidate_profile)
    baseline_payload = _profiles(baseline_summary)[baseline_profile]
    candidate_payload = _profiles(current_summary)[candidate_profile]
    datasets = sorted(set(baseline_payload.keys()) & set(candidate_payload.keys()))
    datasets = [ds for ds in datasets if not str(ds).startswith("_") and isinstance(candidate_payload.get(ds), dict)]
    out: Dict[str, Any] = {
        "schema_version": "candidate-profile-comparison-v1",
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "baseline_profile": baseline_profile,
        "candidate_profile": candidate_profile,
        "datasets": {},
        "decision_summary": {},
    }
    utility_improved = 0
    meaningful_utility_improved = 0
    coverage_regressed = 0
    probe_valid_improved = 0
    probe_invalid_count = 0
    probe_valid_dataset_regressions = 0
    strict_positive_count = 0
    random_gain_count = 0
    notes = []
    for dataset in datasets:
        b = _dataset_row(baseline_payload[dataset])
        c = _dataset_row(candidate_payload[dataset])
        d = _delta(c, b)
        if d["utility"] > 0 or d["strict_min_gain"] > 0:
            utility_improved += 1
        if bool(c.get("probe_valid")) and (
            bool(c.get("selected_beats_random"))
            or bool(c.get("selected_beats_matched"))
            or float(c.get("strict_min_gain") or 0.0) > 0.0
        ):
            meaningful_utility_improved += 1
        if d["coverage"] < -0.01:
            coverage_regressed += 1
        if bool(c.get("probe_valid")) and not bool(b.get("probe_valid")):
            probe_valid_improved += 1
        if not bool(c.get("probe_valid")):
            probe_invalid_count += 1
        if bool(c.get("probe_valid")) and (d["utility"] < 0.0 or d["strict_min_gain"] < 0.0):
            probe_valid_dataset_regressions += 1
        if bool(c.get("selected_beats_matched")) and float(c.get("strict_min_gain") or 0.0) > 0.0:
            strict_positive_count += 1
        if bool(c.get("selected_beats_random")):
            random_gain_count += 1
        if abs(float(d.get("utility") or 0.0)) < float(c.get("max_mde_95") or 0.0):
            notes.append(f"{dataset}: utility delta is below candidate MDE; treat as near-noise-floor.")
        out["datasets"][dataset] = {"baseline": b, "candidate": c, "delta": d}
    promote_candidate = bool(
        coverage_regressed == 0
        and probe_invalid_count == 0
        and probe_valid_dataset_regressions == 0
        and strict_positive_count == len(datasets)
    )
    targeted_followup_candidate = bool(
        coverage_regressed == 0
        and meaningful_utility_improved > 0
        and not promote_candidate
    )
    out["decision_summary"] = {
        "dataset_count": int(len(datasets)),
        "utility_improved_dataset_count": int(utility_improved),
        "meaningful_utility_improved_dataset_count": int(meaningful_utility_improved),
        "coverage_regressed_gt_0_01_dataset_count": int(coverage_regressed),
        "probe_valid_improved_dataset_count": int(probe_valid_improved),
        "probe_invalid_dataset_count": int(probe_invalid_count),
        "probe_valid_dataset_regression_count": int(probe_valid_dataset_regressions),
        "strict_positive_dataset_count": int(strict_positive_count),
        "random_gain_dataset_count": int(random_gain_count),
        "promote_candidate": promote_candidate,
        "targeted_followup_candidate": targeted_followup_candidate,
        "promotion_rule": (
            "Promote globally only when Coverage does not regress >0.01, every dataset has valid probe evidence, "
            "no probe-valid dataset regresses, and every dataset has strict positive multi-matched Utility evidence. "
            "Otherwise keep as candidate or targeted follow-up only."
        ),
        "notes": notes,
    }
    return out


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Candidate Profile Comparison",
        "",
        f"- Baseline profile: `{report['baseline_profile']}`",
        f"- Candidate profile: `{report['candidate_profile']}`",
        f"- Promote candidate: `{report['decision_summary']['promote_candidate']}`",
        f"- Targeted follow-up candidate: `{report['decision_summary']['targeted_followup_candidate']}`",
        f"- Rule: {report['decision_summary']['promotion_rule']}",
        "",
        "| Dataset | dCoverage | dUtility | dStrictMin | Probe valid | > Random | > Matched | Strict status | MDE | Effect/MDE | Evidence tier | Failure reason |",
        "|---|---:|---:|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        c = payload.get("candidate") or {}
        d = payload.get("delta") or {}
        lines.append(
            f"| {dataset} | {float(d.get('coverage') or 0):+.6f} | {float(d.get('utility') or 0):+.6f} | "
            f"{float(d.get('strict_min_gain') or 0):+.6f} | {c.get('probe_valid')} | "
            f"{c.get('selected_beats_random')} | {c.get('selected_beats_matched')} | "
            f"{c.get('strict_status')} | {float(c.get('max_mde_95') or 0):.6f} | "
            f"{float(c.get('min_effect_to_mde_ratio') or 0):+.3f} | "
            f"{c.get('evidence_tier')} | {c.get('failure_reason')} |"
        )
    notes = report.get("decision_summary", {}).get("notes") or []
    if notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare candidate run_summary against saved canonical baseline.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--baseline-profile", default="canonical")
    parser.add_argument("--candidate-profile", default="learnability_rescue_no_anti_collapse")
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_comparison(
        baseline_path=args.baseline,
        current_path=args.current,
        baseline_profile=str(args.baseline_profile),
        candidate_profile=str(args.candidate_profile),
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, args.md_output)
    print(f"[18] comparison json: {args.json_output}", flush=True)
    print(f"[18] comparison md: {args.md_output}", flush=True)
    print(f"[18] promote_candidate={report['decision_summary']['promote_candidate']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
