#!/usr/bin/env python3
"""Build a postmortem for the code-domain confirmatory NLL result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json, sha256_file


DEFAULT_DEVELOPMENT_REPORT = OUTPUT_DIR / "validation" / "code_domain_development_decision_report.json"
DEFAULT_CONFIRMATORY_REPORT = OUTPUT_DIR / "validation" / "code_domain_confirmatory_decision_report.json"
DEFAULT_DEVELOPMENT_HELDOUT = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1" / "heldouts" / "development_code_nll_heldout.jsonl"
DEFAULT_CONFIRMATORY_HELDOUT = OUTPUT_DIR / "code_domain_confirmatory_qwen3_4b_v1" / "heldouts" / "confirmatory_code_nll_heldout.jsonl"
DEFAULT_ARMS_DIR = OUTPUT_DIR / "temporal_code_training_freeze_v1" / "equal_token_arms"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_confirmatory_postmortem_report.json"
DEFAULT_DOC = Path("docs") / "code_domain_confirmatory_postmortem.md"

TRAINED_ARMS = (
    "raw_random_equal_budget",
    "stageA_random_equal_budget",
    "curated_equal_budget",
    "known_high_quality_equal_budget",
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _paired_ci(values: List[float]) -> Dict[str, Any]:
    mean = _mean(values)
    std = _std(values)
    se = std / math.sqrt(len(values)) if values else math.inf
    # t critical for df=4, two-sided 95%. This is descriptive, not a new gate.
    t_crit = 2.776 if len(values) == 5 else 1.96
    return {
        "mean": mean,
        "sample_std": std,
        "standard_error": se,
        "descriptive_95_ci": [mean - t_crit * se, mean + t_crit * se],
    }


def _jsonl_profile(path: Path) -> Dict[str, Any]:
    rows = list(iter_jsonl_records_resilient(path))
    token_counts = [int(row.get("token_proxy_count") or row.get("token_proxy") or 0) for row in rows]
    repos = Counter(str(row.get("repository_identity") or row.get("repository") or row.get("repo") or "missing") for row in rows)
    content_types = Counter(str(row.get("content_type") or "missing") for row in rows)
    chunk_kinds = Counter(str(row.get("chunk_kind") or "missing") for row in rows)
    change_types = Counter(str(row.get("change_type") or "missing") for row in rows)
    path_suffixes = Counter(Path(str(row.get("path") or "")).suffix.lower() or "no_suffix" for row in rows)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "records": len(rows),
        "token_proxy_sum": sum(token_counts),
        "token_proxy_mean": _mean(token_counts) if token_counts else 0.0,
        "token_proxy_median": statistics.median(token_counts) if token_counts else 0.0,
        "content_type_counts": dict(content_types),
        "content_type_ratios": {
            key: value / len(rows) for key, value in sorted(content_types.items())
        } if rows else {},
        "chunk_kind_counts": dict(chunk_kinds),
        "change_type_counts": dict(change_types),
        "path_suffix_counts": dict(path_suffixes),
        "repository_count": len(repos),
        "top_repositories": repos.most_common(12),
        "repository_set": sorted(repos),
    }


def _arm_nll_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = report["summary"]
    base = float(summary["base_no_update_mean_nll"])
    arms = summary["arm_summaries"]
    rows: Dict[str, Any] = {
        "base_no_update": {
            "mean_nll": base,
            "improvement_vs_base": 0.0,
            "relative_improvement_vs_base": 0.0,
        }
    }
    for arm, arm_summary in arms.items():
        mean_nll = float(arm_summary["mean_nll"])
        improvement = base - mean_nll
        rows[arm] = {
            "mean_nll": mean_nll,
            "sample_std_nll": arm_summary["sample_std_nll"],
            "improvement_vs_base": improvement,
            "relative_improvement_vs_base": improvement / base,
            "per_seed": arm_summary["per_seed"],
        }
    return rows


def _delta_summary(report: Dict[str, Any], key: str, margin: float | None = None) -> Dict[str, Any]:
    row = report["summary"]["nll_gate"]["paired_deltas"][key]
    values = [float(value) for value in row["per_seed_delta"].values()]
    result = {
        "label": row["label"],
        "per_seed_delta": row["per_seed_delta"],
        "all_seed_deltas_positive": row["all_seed_deltas_positive"],
        **_paired_ci(values),
    }
    if margin is not None:
        result["margin"] = margin
        result["gap_to_margin"] = margin - result["mean"]
        result["descriptive_ci_excludes_margin"] = result["descriptive_95_ci"][1] < margin
    return result


def _effect_shift(dev: Dict[str, Any], con: Dict[str, Any]) -> Dict[str, Any]:
    dev_gate = dev["summary"]["nll_gate"]
    con_gate = con["summary"]["nll_gate"]
    dev_primary = float(dev_gate["curated_vs_stageA_random_mean_nll_reduction"])
    con_primary = float(con_gate["curated_vs_stageA_random_mean_nll_reduction"])
    dev_raw = float(dev_gate["curated_vs_raw_random_mean_nll_reduction"])
    con_raw = float(con_gate["curated_vs_raw_random_mean_nll_reduction"])
    margin = float(con_gate["primary_margin_required_absolute_nll_reduction"])
    return {
        "primary_stageA_minus_curated": {
            "development": dev_primary,
            "confirmatory": con_primary,
            "absolute_shrink": dev_primary - con_primary,
            "confirmatory_retention_ratio": con_primary / dev_primary,
            "margin": margin,
            "confirmatory_gap_to_margin": margin - con_primary,
        },
        "raw_random_minus_curated": {
            "development": dev_raw,
            "confirmatory": con_raw,
            "absolute_shrink": dev_raw - con_raw,
            "confirmatory_retention_ratio": con_raw / dev_raw,
        },
        "base_nll_scale": {
            "development_base_nll": float(dev["summary"]["base_no_update_mean_nll"]),
            "confirmatory_base_nll": float(con["summary"]["base_no_update_mean_nll"]),
            "confirmatory_minus_development_base_nll": (
                float(con["summary"]["base_no_update_mean_nll"])
                - float(dev["summary"]["base_no_update_mean_nll"])
            ),
            "absolute_margin_as_fraction_of_development_base_nll": margin / float(dev["summary"]["base_no_update_mean_nll"]),
            "absolute_margin_as_fraction_of_confirmatory_base_nll": margin / float(con["summary"]["base_no_update_mean_nll"]),
        },
    }


def _heldout_shift(dev_profile: Dict[str, Any], con_profile: Dict[str, Any]) -> Dict[str, Any]:
    dev_repos = set(dev_profile["repository_set"])
    con_repos = set(con_profile["repository_set"])
    dev_test_ratio = dev_profile["content_type_ratios"].get("test", 0.0)
    con_test_ratio = con_profile["content_type_ratios"].get("test", 0.0)
    return {
        "record_count_change": con_profile["records"] - dev_profile["records"],
        "token_proxy_sum_change": con_profile["token_proxy_sum"] - dev_profile["token_proxy_sum"],
        "repository_count_change": con_profile["repository_count"] - dev_profile["repository_count"],
        "repository_intersection": sorted(dev_repos & con_repos),
        "repository_jaccard": len(dev_repos & con_repos) / len(dev_repos | con_repos) if (dev_repos | con_repos) else 0.0,
        "test_ratio_development": dev_test_ratio,
        "test_ratio_confirmatory": con_test_ratio,
        "test_ratio_increase": con_test_ratio - dev_test_ratio,
        "code_ratio_development": dev_profile["content_type_ratios"].get("code", 0.0),
        "code_ratio_confirmatory": con_profile["content_type_ratios"].get("code", 0.0),
        "median_token_proxy_change": con_profile["token_proxy_median"] - dev_profile["token_proxy_median"],
        "mean_token_proxy_change": con_profile["token_proxy_mean"] - dev_profile["token_proxy_mean"],
    }


def _decision_implications(confirmatory: Dict[str, Any]) -> Dict[str, Any]:
    gate = confirmatory["summary"]["nll_gate"]
    return {
        "frozen_confirmatory_result_locked": True,
        "status": confirmatory["status"],
        "primary_margin_passed": bool(gate["curated_vs_stageA_random_margin_pass"]),
        "directional_stageA_signal_replicated": bool(
            gate["paired_deltas"]["stageA_random_minus_curated"]["all_seed_deltas_positive"]
        ),
        "directional_raw_signal_replicated": bool(
            gate["paired_deltas"]["raw_random_minus_curated"]["all_seed_deltas_positive"]
        ),
        "claim": (
            "negative_confirmatory_primary_margin_result_with_positive_directional_signal"
            if confirmatory["status"] == "confirmatory_decision_reject_primary_margin_failure"
            else "see_confirmatory_decision_report"
        ),
        "forbidden_response": [
            "do_not_change_the_frozen_margin_after_confirmatory_outcomes",
            "do_not_change_confirmatory_seeds_or_heldout_after_outcomes",
            "do_not_move_utility_or_benchmark_outcomes_into_stage_b",
            "do_not_reinterpret_the_failed_margin_as_a_pass",
        ],
    }


def _next_cycle_requirements() -> Dict[str, Any]:
    return {
        "new_cycle_required": True,
        "must_remain_separate_from_completed_confirmatory_protocol": True,
        "recommended_changes": [
            {
                "area": "heldout_design",
                "action": "freeze larger and stratified development/confirmatory heldouts by repository, content_type, and code/test ratio",
                "reason": "The completed confirmatory split had fewer repositories and a much higher test ratio than development.",
            },
            {
                "area": "margin_calibration",
                "action": "predeclare margins from development-only power/effect-size analysis and consider relative or stratified NLL margins",
                "reason": "The absolute 0.005 margin was not reached after the confirmatory base NLL scale shifted lower.",
            },
            {
                "area": "stage_b_selector",
                "action": "strengthen code-quality proxy selection using Stage-A/development-only evidence, with no Utility or benchmark leakage into Stage B",
                "reason": "Curated remained directional but did not reach known-high-quality mean NLL.",
            },
            {
                "area": "baseline_diagnostics",
                "action": "track Stage-A-random hardness separately by split before freezing confirmatory margins",
                "reason": "The Stage-A-random to curated gap compressed sharply in confirmatory.",
            },
            {
                "area": "reporting",
                "action": "report this cycle as a negative confirmatory result, not as a failed run",
                "reason": "All training and NLL evaluations completed under the frozen protocol.",
            },
        ],
    }


def _build_markdown(report: Dict[str, Any]) -> str:
    eff = report["effect_shift"]
    con = report["confirmatory_result"]
    held = report["heldout_shift"]
    lines = [
        "# Code-Domain Confirmatory Postmortem",
        "",
        "## Status",
        "",
        f"- Confirmatory status: `{con['status']}`.",
        "- Interpretation: negative primary-margin result with a positive directional curation signal.",
        "- This is a completed confirmatory experiment, not an infrastructure failure.",
        "",
        "## Primary NLL Finding",
        "",
        f"- Required frozen margin: `{con['frozen_margin']}`.",
        f"- Curated vs Stage-A-random reduction: `{con['curated_vs_stageA_random_reduction']}`.",
        f"- Gap to margin: `{con['gap_to_margin']}`.",
        f"- Curated vs raw-random reduction: `{con['curated_vs_raw_random_reduction']}`.",
        f"- Known-HQ minus curated: `{con['known_high_quality_minus_curated']}`.",
        "",
        "## Development To Confirmatory Shift",
        "",
        f"- Development primary reduction: `{eff['primary_stageA_minus_curated']['development']}`.",
        f"- Confirmatory primary reduction: `{eff['primary_stageA_minus_curated']['confirmatory']}`.",
        f"- Retention ratio: `{eff['primary_stageA_minus_curated']['confirmatory_retention_ratio']}`.",
        f"- Absolute shrink: `{eff['primary_stageA_minus_curated']['absolute_shrink']}`.",
        f"- Development base NLL: `{eff['base_nll_scale']['development_base_nll']}`.",
        f"- Confirmatory base NLL: `{eff['base_nll_scale']['confirmatory_base_nll']}`.",
        "",
        "## Heldout Shift",
        "",
        f"- Record count change: `{held['record_count_change']}`.",
        f"- Repository Jaccard overlap: `{held['repository_jaccard']}`.",
        f"- Test ratio development: `{held['test_ratio_development']}`.",
        f"- Test ratio confirmatory: `{held['test_ratio_confirmatory']}`.",
        f"- Test ratio increase: `{held['test_ratio_increase']}`.",
        f"- Mean token-proxy change: `{held['mean_token_proxy_change']}`.",
        "",
        "## Locked Interpretation",
        "",
        "- The completed frozen confirmatory protocol must remain negative on the primary margin.",
        "- Margins, seeds, heldout slices, token budgets, and Stage-C thresholds must not be changed post hoc.",
        "- Utility, EvalPlus, and retention outcomes remain Stage C only and must not enter Stage B selector objectives.",
        "",
        "## Next Development Cycle",
        "",
        "- Start a new development cycle if improving the recipe.",
        "- Freeze larger stratified heldouts by repository and content type.",
        "- Calibrate the practical margin before confirmatory outcomes using development-only power/effect-size analysis.",
        "- Strengthen Stage B proxy selection without Utility or benchmark leakage.",
        "- Treat this result as valid negative evidence in the paper trail.",
        "",
    ]
    return "\n".join(lines)


def build(
    development_report_path: Path,
    confirmatory_report_path: Path,
    development_heldout_path: Path,
    confirmatory_heldout_path: Path,
    arms_dir: Path,
    output_path: Path,
    doc_path: Path,
) -> Dict[str, Any]:
    development = load_json(development_report_path)
    confirmatory = load_json(confirmatory_report_path)
    margin = float(confirmatory["summary"]["nll_gate"]["primary_margin_required_absolute_nll_reduction"])
    dev_profile = _jsonl_profile(development_heldout_path)
    con_profile = _jsonl_profile(confirmatory_heldout_path)
    arm_payload_profiles = {
        arm: _jsonl_profile(arms_dir / f"{arm}.jsonl")
        for arm in TRAINED_ARMS
    }
    effect_shift = _effect_shift(development, confirmatory)
    confirmatory_primary = _delta_summary(confirmatory, "stageA_random_minus_curated", margin)
    report = {
        "schema_version": "code-domain-confirmatory-postmortem-v1",
        "status": "confirmatory_postmortem_completed",
        "source_sha256": {
            str(development_report_path): sha256_file(development_report_path),
            str(confirmatory_report_path): sha256_file(confirmatory_report_path),
            str(development_heldout_path): sha256_file(development_heldout_path),
            str(confirmatory_heldout_path): sha256_file(confirmatory_heldout_path),
        },
        "confirmatory_result": {
            "status": confirmatory["status"],
            "training_runs_completed": confirmatory["summary"]["training_runs_completed"],
            "heldout_nll_results_completed": confirmatory["summary"]["heldout_nll_results_completed"],
            "frozen_margin": margin,
            "curated_vs_stageA_random_reduction": confirmatory["summary"]["nll_gate"][
                "curated_vs_stageA_random_mean_nll_reduction"
            ],
            "gap_to_margin": margin - confirmatory["summary"]["nll_gate"][
                "curated_vs_stageA_random_mean_nll_reduction"
            ],
            "curated_vs_raw_random_reduction": confirmatory["summary"]["nll_gate"][
                "curated_vs_raw_random_mean_nll_reduction"
            ],
            "known_high_quality_minus_curated": confirmatory["summary"]["nll_gate"][
                "known_high_quality_minus_curated_mean_nll"
            ],
            "primary_delta_descriptive_stats": confirmatory_primary,
        },
        "development_result": {
            "status": development["status"],
            "curated_vs_stageA_random_reduction": development["summary"]["nll_gate"][
                "curated_vs_stageA_random_mean_nll_reduction"
            ],
            "curated_vs_raw_random_reduction": development["summary"]["nll_gate"][
                "curated_vs_raw_random_mean_nll_reduction"
            ],
        },
        "development_nll_by_arm": _arm_nll_summary(development),
        "confirmatory_nll_by_arm": _arm_nll_summary(confirmatory),
        "effect_shift": effect_shift,
        "heldout_profiles": {
            "development": dev_profile,
            "confirmatory": con_profile,
        },
        "heldout_shift": _heldout_shift(dev_profile, con_profile),
        "training_payload_profiles": arm_payload_profiles,
        "diagnosis": {
            "primary_cause": "confirmatory_effect_size_below_predeclared_absolute_margin",
            "supporting_observations": [
                "directional curated improvement replicated against Stage-A-random and raw-random on every confirmatory seed",
                "development-to-confirmatory primary effect retained only about one third of its development magnitude",
                "confirmatory heldout had lower base NLL, fewer repositories, and a higher test ratio than development",
                "Stage-A-random baseline compressed toward curated on the confirmatory split",
                "known-high-quality remained slightly lower NLL than curated",
            ],
            "not_supported_claims": [
                "the frozen confirmatory recipe passed its primary margin",
                "Stage-C guardrails can rescue the primary-margin failure",
                "the margin can be changed post hoc",
            ],
        },
        "decision_implications": _decision_implications(confirmatory),
        "next_development_cycle": _next_cycle_requirements(),
        "confirmatory_outcomes_read": True,
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": (
            "Postmortem of completed frozen confirmatory outcomes. This report must not "
            "be used to retrofit the completed confirmatory protocol; it can only inform "
            "a separate future development cycle."
        ),
    }
    save_json(output_path, report)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(_build_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain confirmatory postmortem report.")
    parser.add_argument("--development-report", type=Path, default=DEFAULT_DEVELOPMENT_REPORT)
    parser.add_argument("--confirmatory-report", type=Path, default=DEFAULT_CONFIRMATORY_REPORT)
    parser.add_argument("--development-heldout", type=Path, default=DEFAULT_DEVELOPMENT_HELDOUT)
    parser.add_argument("--confirmatory-heldout", type=Path, default=DEFAULT_CONFIRMATORY_HELDOUT)
    parser.add_argument("--arms-dir", type=Path, default=DEFAULT_ARMS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    report = build(
        args.development_report,
        args.confirmatory_report,
        args.development_heldout,
        args.confirmatory_heldout,
        args.arms_dir,
        args.output,
        args.doc,
    )
    print(
        {
            "status": report["status"],
            "confirmatory_status": report["confirmatory_result"]["status"],
            "gap_to_margin": report["confirmatory_result"]["gap_to_margin"],
            "effect_retention_ratio": report["effect_shift"]["primary_stageA_minus_curated"][
                "confirmatory_retention_ratio"
            ],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
