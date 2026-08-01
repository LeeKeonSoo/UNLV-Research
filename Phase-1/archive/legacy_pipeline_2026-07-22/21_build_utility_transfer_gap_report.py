#!/usr/bin/env python3
"""Build a report for feature-space gains that do not transfer to LM Utility."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, load_json, save_json


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.md"
SELECTOR_BASELINE_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"
ANTI_MEMORIZATION_PROBE_REPORT_PATH = OUTPUT_DIR / "validation" / "anti_memorization_probe_report.json"
ANTI_MEMORIZATION_PROBE_REPORT_GLOB = "anti_memorization_probe_report*.json"
UTILITY_POWER_SWEEP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_probe_power_sweep.json"
ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE = "baseline_anti_memorization_matched_stageA_random"
REPLICATE_PRESET_RE = re.compile(r"^(?P<family>.+)_b(?P<replicate>\d+)$")


def _profile_payload(run_summary: Dict[str, Any], profile: str) -> Dict[str, Any]:
    profiles = run_summary.get("profiles") or {}
    if profile in profiles and isinstance(profiles[profile], dict):
        return profiles[profile]
    names = [str(name) for name, payload in profiles.items() if not str(name).startswith("_") and isinstance(payload, dict)]
    if len(names) == 1:
        return profiles[names[0]]
    raise RuntimeError(f"Profile {profile!r} not found. Available profiles: {names}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _utility_aggregate(meta: Dict[str, Any]) -> Dict[str, Any]:
    return ((meta.get("utility_probe_details") or {}).get("aggregate") or {})


def _utility_evidence(meta: Dict[str, Any]) -> Dict[str, Any]:
    return (_utility_aggregate(meta).get("utility_evidence_summary") or {})


def _selector_comparisons(selector_audit: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    payload = ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
    comparisons = payload.get("comparisons") if isinstance(payload, dict) else None
    return comparisons if isinstance(comparisons, dict) else {}


def _verdict_summary(comparisons: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    verdict = ((comparisons.get(baseline) or {}).get("verdict") or {})
    if not isinstance(verdict, dict):
        verdict = {}
    return {
        "verdict": verdict.get("verdict"),
        "quality_delta": verdict.get("quality_delta"),
        "learnability_delta": verdict.get("learnability_delta"),
        "redundancy_risk_delta": verdict.get("redundancy_risk_delta"),
        "word_count_delta": verdict.get("word_count_delta"),
    }


def _margin_status(delta: Any, mde: Any) -> Dict[str, Any]:
    delta_f = _safe_float(delta)
    mde_f = abs(_safe_float(mde))
    return {
        "delta_nll": round(delta_f, 8),
        "mde": round(mde_f, 8),
        "positive": bool(delta_f > 0.0),
        "decisive": bool(mde_f > 0.0 and abs(delta_f) > mde_f),
        "near_noise_floor": bool(mde_f > 0.0 and abs(delta_f) <= mde_f),
    }


def _load_anti_memorization_reports() -> Dict[str, Any]:
    reports: List[Dict[str, Any]] = []
    paths: List[str] = []
    validation_dir = OUTPUT_DIR / "validation"
    for path in sorted(validation_dir.glob(ANTI_MEMORIZATION_PROBE_REPORT_GLOB)):
        try:
            report = load_json(path)
        except Exception:
            continue
        if not isinstance(report, dict):
            continue
        if report.get("schema_version") != "anti-memorization-probe-report-v1":
            continue
        reports.append(report)
        paths.append(str(path))
    return {"reports": reports, "paths": paths}


def _anti_memorization_probe_result(report_bundle: Dict[str, Any], dataset: str, profile: str) -> Dict[str, Any]:
    if not isinstance(report_bundle, dict):
        return {}
    reports = report_bundle.get("reports")
    if not isinstance(reports, list):
        reports = [report_bundle]
    for report in reports:
        if not isinstance(report, dict):
            continue
        if str(report.get("dataset") or "") != str(dataset):
            continue
        if str(report.get("profile") or "") != str(profile):
            continue
        utility = report.get("utility_result")
        return utility if isinstance(utility, dict) else {}
    return {}


def _anti_memorization_support(result: Dict[str, Any]) -> Dict[str, Any]:
    delta = _safe_float(result.get("delta_nll"))
    ci_low = _safe_float(result.get("delta_nll_ci_low"))
    mde = _safe_float(result.get("minimum_detectable_delta_nll_95_max"))
    effect_to_mde = _safe_float(result.get("effect_to_mde_ratio_min"))
    causal = result.get("causal_utility_audit") or {}
    return {
        "available": bool(result),
        "baseline": ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE if result else None,
        "supports_selected": bool(delta > 0.0 and ci_low > 0.0 and (mde <= 0.0 or abs(delta) > mde)),
        "delta_nll": round(delta, 8),
        "delta_nll_ci_low": round(ci_low, 8),
        "minimum_detectable_delta_nll_95_max": round(mde, 8),
        "effect_to_mde_ratio_min": round(effect_to_mde, 6),
        "detectable_effect_fraction": result.get("detectable_effect_fraction"),
        "small_lm_probe_gain_score": result.get("small_lm_probe_gain_score"),
        "causal_mode": causal.get("dominant_failure_mode"),
        "train_audit_gap": causal.get("mean_selected_minus_baseline_train_audit_delta_nll"),
    }


def _replicate_status_by_family(runs: Dict[str, Any]) -> Dict[str, Dict[int, bool]]:
    status_by_family: Dict[str, Dict[int, bool]] = {}
    for preset, run in runs.items():
        if not isinstance(run, dict) or not run.get("exists") or not run.get("compatible"):
            continue
        match = REPLICATE_PRESET_RE.match(str(preset))
        if not match:
            continue
        family = match.group("family")
        replicate = int(match.group("replicate"))
        status_by_family.setdefault(family, {})[replicate] = bool(
            run.get("probe_valid") and run.get("selected_gt_random")
        )
    return status_by_family


def _replicated_family_replicates(runs: Dict[str, Any]) -> Dict[str, List[int]]:
    status_by_family = _replicate_status_by_family(runs)
    return {
        family: sorted(status_by_replicate)
        for family, status_by_replicate in sorted(status_by_family.items())
        if len(status_by_replicate) >= 2 and all(status_by_replicate.values())
    }


def _best_replicated_preset(runs: Dict[str, Any], replicated: Dict[str, List[int]]) -> str | None:
    candidates = [
        f"{family}_b{replicate}"
        for family, replicates in replicated.items()
        for replicate in replicates
        if isinstance((runs.get(f"{family}_b{replicate}") or {}), dict)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (
            _safe_float((runs.get(name) or {}).get("selected_minus_random")),
            str(name),
        ),
    )


def _power_sweep_summary(report: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    payload = ((report.get("datasets") or {}).get(str(dataset)) or {})
    if not isinstance(payload, dict):
        return {}
    decision = payload.get("decision") or {}
    runs = payload.get("runs") or {}
    compatible_runs = [
        run
        for run in runs.values()
        if isinstance(run, dict) and bool(run.get("exists")) and bool(run.get("compatible"))
    ]
    selected_positive_runs = [run for run in compatible_runs if bool(run.get("selected_gt_random"))]
    valid_runs = [run for run in compatible_runs if bool(run.get("probe_valid"))]
    valid_selected_positive_runs = [
        run
        for run in compatible_runs
        if bool(run.get("probe_valid")) and bool(run.get("selected_gt_random"))
    ]
    valid_selected_positive_presets = [
        str(name)
        for name, run in runs.items()
        if isinstance(run, dict)
        and bool(run.get("exists"))
        and bool(run.get("compatible"))
        and bool(run.get("probe_valid"))
        and bool(run.get("selected_gt_random"))
    ]
    best_valid_selected_preset = None
    if valid_selected_positive_presets:
        best_valid_selected_preset = max(
            valid_selected_positive_presets,
            key=lambda name: (
                _safe_float((runs.get(name) or {}).get("selected_minus_random")),
                str(name),
            ),
        )
    hash_noise_valid_runs = [
        run
        for run in compatible_runs
        if bool(run.get("probe_valid")) and str(run.get("corruption_mode") or "") == "hash_noise"
    ]
    replicated = _replicated_family_replicates(runs)
    replicated_families = sorted(replicated.keys())
    best_replicated_preset = _best_replicated_preset(runs, replicated)
    return {
        "available": bool(compatible_runs),
        "existing_runs": int(decision.get("existing_runs") or 0),
        "compatible_runs": int(decision.get("compatible_runs") or len(compatible_runs)),
        "probe_valid_runs": int(decision.get("probe_valid_runs") or len(valid_runs)),
        "selected_gt_random_runs": int(decision.get("selected_gt_random_runs") or len(selected_positive_runs)),
        "valid_selected_gt_random_runs": int(len(valid_selected_positive_runs)),
        "valid_selected_gt_random_presets": valid_selected_positive_presets,
        "best_valid_selected_gt_random_preset": best_valid_selected_preset,
        "replicated_valid_families": replicated_families,
        "replicated_valid_family_replicates": replicated,
        "best_replicated_valid_preset": best_replicated_preset,
        "best_replicated_valid_family": (
            REPLICATE_PRESET_RE.match(best_replicated_preset).group("family")
            if best_replicated_preset and REPLICATE_PRESET_RE.match(best_replicated_preset)
            else None
        ),
        "hash_noise_valid_runs": int(len(hash_noise_valid_runs)),
        "stable_probe_valid": bool(decision.get("stable_probe_valid")),
        "recommended_next_action": (
            f"Replicated valid selected>Stage-A-random family `{REPLICATE_PRESET_RE.match(best_replicated_preset).group('family')}` exists; standardize this Stage-C protocol candidate before selector/Core tuning."
            if best_replicated_preset and REPLICATE_PRESET_RE.match(best_replicated_preset)
            else decision.get("recommended_next_action")
        ),
        "selected_not_supported_across_sweep": bool(compatible_runs and not selected_positive_runs),
        "selected_not_supported_in_valid_runs": bool(valid_runs and not valid_selected_positive_runs),
    }


def _matched_memorization_proxy_gap(feature_matched: Dict[str, Any], comparisons: Dict[str, Any]) -> Dict[str, Any]:
    numeric = ((comparisons.get("multi_matched_stageA_random") or {}).get("numeric_comparison") or {})
    repeat_delta = _safe_float((numeric.get("intra_chunk_repeat_pressure") or {}).get("delta_selected_minus_baseline"))
    word_count_delta = _safe_float((numeric.get("word_count") or {}).get("delta_selected_minus_baseline"))
    useful_recurrence_delta = _safe_float((numeric.get("useful_recurrence_score") or {}).get("delta_selected_minus_baseline"))
    quality_delta = _safe_float(feature_matched.get("quality_delta"))
    redundancy_risk_delta = _safe_float(feature_matched.get("redundancy_risk_delta"))
    lexical_diversity_delta = _safe_float((numeric.get("lexical_diversity") or {}).get("delta_selected_minus_baseline"))
    warning_delta = _safe_float((numeric.get("validity_warning_count") or {}).get("delta_selected_minus_baseline"))
    return {
        "baseline_more_repetitive": bool(repeat_delta <= -0.05),
        "baseline_longer": bool(word_count_delta <= -10.0),
        "selected_higher_quality": bool(quality_delta > 0.0),
        "selected_lower_redundancy_risk": bool(redundancy_risk_delta < 0.0),
        "selected_higher_lexical_diversity": bool(lexical_diversity_delta > 0.0),
        "selected_fewer_validity_warnings": bool(warning_delta < 0.0),
        "repeat_pressure_delta": round(repeat_delta, 6),
        "word_count_delta": round(word_count_delta, 6),
        "useful_recurrence_delta": round(useful_recurrence_delta, 6),
        "quality_delta": round(quality_delta, 6),
        "redundancy_risk_delta": round(redundancy_risk_delta, 6),
        "lexical_diversity_delta": round(lexical_diversity_delta, 6),
        "validity_warning_delta": round(warning_delta, 6),
    }


def _framework_implication(
    *,
    category: str,
    stage_c_pass: bool,
    probe_valid: bool,
    feature_stronger: bool,
    curation_margin: Dict[str, Any],
    power: Dict[str, Any],
    anti_memorization_support: Dict[str, Any],
    token_exposure_caveat: bool,
) -> Dict[str, Any]:
    if stage_c_pass:
        if token_exposure_caveat:
            return {
                "status": "stage_c_development_ready_with_token_exposure_caveat",
                "selector_policy_action": "hold",
                "strict_baseline_action": "certification_followup",
                "interpretation": (
                    "The current subset passes Stage-C development validation, but token-inventory stress is not cleanly separated. "
                    "Keep Utility as validation-only evidence and rerun certification-grade Stage C before promotion."
                ),
            }
        return {
            "status": "stage_c_development_ready",
            "selector_policy_action": "hold",
            "strict_baseline_action": "certification_followup",
            "interpretation": (
                "The current subset passes Stage-C development validation. Keep Utility as "
                "validation-only evidence and run certification-grade follow-up before promotion."
            ),
        }
    if category == "probe_preset_candidate_available":
        return {
            "status": "utility_probe_preset_instability",
            "selector_policy_action": "hold",
            "strict_baseline_action": "standardize_probe_preset",
            "interpretation": (
                "The default Stage-C sensitivity status is not interpretable, but power sweep "
                "contains valid selected>random preset candidates. Standardize and rerun the "
                "probe protocol before treating Utility as selector evidence."
            ),
        }
    if not probe_valid:
        return {
            "status": "utility_probe_not_interpretable",
            "selector_policy_action": "hold",
            "strict_baseline_action": "hold",
            "interpretation": (
                "Stage-B feature gains must not be treated as Utility failures until the "
                "small-LM probe separates positive, Stage-A random, and destructive controls."
            ),
        }
    if category == "utility_power_sweep_selected_not_supported":
        return {
            "status": "core_policy_proxy_not_utility_supported",
            "selector_policy_action": "inspect_core_policy_proxy",
            "strict_baseline_action": "hold",
            "interpretation": (
                "The selector is stronger in Core feature space, but valid Utility sweep arms do "
                "not support selected over Stage-A random. Inspect Quality/Redundancy/Learnability "
                "proxy calibration without adding Utility to the selector objective."
            ),
        }
    if category == "anti_memorization_probe_supports_selector":
        return {
            "status": "strict_baseline_confounded_by_easy_nll_signal",
            "selector_policy_action": "hold",
            "strict_baseline_action": "add_or_promote_repeat_pressure_control",
            "interpretation": (
                "The selected subset loses to the canonical strict baseline but beats a "
                "repeat-pressure-matched diagnostic. Treat the strict baseline as a confound "
                "candidate before blaming Stage-B selection."
            ),
        }
    if category == "lm_train_memorization_proxy_gap":
        return {
            "status": "possible_easy_nll_baseline_confound",
            "selector_policy_action": "hold",
            "strict_baseline_action": "run_repeat_pressure_matched_diagnostic",
            "interpretation": (
                "The matched baseline appears easier for the probe to learn because of length or "
                "repetition pressure. Run anti-memorization controls before changing the selector."
            ),
        }
    if feature_stronger and not curation_margin.get("positive"):
        return {
            "status": "feature_utility_transfer_gap",
            "selector_policy_action": "diagnose",
            "strict_baseline_action": "hold",
            "interpretation": (
                "Core feature gains do not transfer to Stage-C curation benefit under the current "
                "probe. Inspect train/eval alignment and Core proxy calibration."
            ),
        }
    if power.get("stable_probe_valid") and anti_memorization_support.get("supports_selected"):
        return {
            "status": "candidate_ready_for_strict_protocol_revision",
            "selector_policy_action": "hold",
            "strict_baseline_action": "integrate_diagnostic_as_reported_control",
            "interpretation": (
                "Utility diagnostics support the selected subset, but certification still needs an "
                "explicit strict-baseline protocol decision."
            ),
        }
    return {
        "status": "strict_counterfactual_not_resolved",
        "selector_policy_action": "hold",
        "strict_baseline_action": "inspect",
        "interpretation": (
            "The dataset is not Stage-C ready under the current strict counterfactual protocol."
        ),
    }


def _classify_transfer_gap(
    *,
    stage_c_pass: bool,
    feature_random: Dict[str, Any],
    feature_matched: Dict[str, Any],
    comparisons: Dict[str, Any],
    evidence: Dict[str, Any],
    aggregate: Dict[str, Any],
    anti_memorization_probe: Dict[str, Any],
    power_sweep: Dict[str, Any],
) -> Dict[str, Any]:
    curation = evidence.get("curation_benefit_status") or aggregate.get("curation_benefit_status") or {}
    strict = evidence.get("strict_counterfactual_status") or aggregate.get("strict_counterfactual_status") or {}
    causal = aggregate.get("causal_utility_audit") or evidence.get("causal_utility_audit") or {}
    failure_analysis = aggregate.get("utility_failure_analysis") or {}
    learning_signal = failure_analysis.get("learning_signal_coverage_diagnostic") or {}
    matched_baseline_deltas = failure_analysis.get("matched_baseline_deltas") or {}
    anti_memorization_result = matched_baseline_deltas.get(ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE) or {}
    anti_memorization_support = _anti_memorization_support(anti_memorization_probe or anti_memorization_result)
    power = _power_sweep_summary(power_sweep, str(aggregate.get("in_domain_dataset") or ""))
    probe = evidence.get("probe_sensitivity_status") or aggregate.get("probe_sensitivity_status") or {}
    token_exposure_caveat = bool(
        probe.get("token_exposure_confounded") or probe.get("token_exposure_inconclusive")
    )
    feature_stronger = str(feature_random.get("verdict") or "") == "selected_meaningfully_stronger"
    probe_valid = bool(probe.get("destructive_probe_valid", probe.get("probe_valid")))
    curation_margin = _margin_status(curation.get("delta_nll"), curation.get("minimum_detectable_delta_nll_95"))
    strict_delta = _safe_float(strict.get("min_delta_nll"))
    strict_effect_to_mde = _safe_float(strict.get("min_effect_to_mde_ratio"))
    strict_mde = abs(strict_delta / strict_effect_to_mde) if strict_effect_to_mde else 0.0
    strict_margin = _margin_status(strict_delta, strict_mde)
    train_gap = _safe_float(causal.get("mean_selected_minus_baseline_train_audit_delta_nll"))
    causal_mode = str(evidence.get("causal_failure_mode") or causal.get("dominant_failure_mode") or "")
    memorization_gap = _matched_memorization_proxy_gap(feature_matched, comparisons)
    easy_nll_tension = bool(
        memorization_gap["selected_higher_quality"]
        and memorization_gap["selected_lower_redundancy_risk"]
        and (memorization_gap["baseline_more_repetitive"] or memorization_gap["baseline_longer"])
    )
    learning_gaps = learning_signal.get("gaps_selected_minus_baseline") or {}
    template_density_gap = _safe_float(learning_gaps.get("template_density"))
    moderate_difficulty_gap = _safe_float(learning_gaps.get("moderate_difficulty_share"))
    phrase_novelty_gap = _safe_float(learning_gaps.get("unique_bigram_ratio"))
    memorization_like = bool(
        train_gap < -0.001
        and memorization_gap["selected_higher_quality"]
        and memorization_gap["selected_lower_redundancy_risk"]
        and (memorization_gap["baseline_more_repetitive"] or memorization_gap["baseline_longer"])
        and (
            not learning_gaps
            or template_density_gap < -0.05
            or moderate_difficulty_gap > 0.05
            or phrase_novelty_gap >= 0.0
        )
    )

    if stage_c_pass:
        if token_exposure_caveat:
            category = "stage_c_development_ready_with_token_exposure_caveat"
            action = (
                "Current Stage-C development validation passes, but token-inventory stress is inconclusive; "
                "keep as targeted follow-up and rerun certification-grade Utility before promotion."
            )
        else:
            category = "stage_c_development_ready"
            action = "Current Stage-C development validation passes; keep as targeted follow-up and run certification-grade Utility before promotion."
    elif not feature_stronger:
        category = "selector_feature_space_not_stronger"
        action = "Improve Stage-B feature-space selection before Utility transfer analysis."
    elif not probe_valid and power.get("valid_selected_gt_random_runs", 0) > 0:
        category = "probe_preset_candidate_available"
        action = (
            "Power sweep found valid selected>Stage-A-random probe presets; standardize a stronger "
            "Utility preset and rerun Stage-C before selector/Core tuning."
        )
    elif not probe_valid:
        category = "probe_not_ready_for_transfer_claim"
        action = "Increase probe/control power before interpreting feature-to-Utility transfer."
    elif power.get("selected_not_supported_across_sweep") and power.get("selected_not_supported_in_valid_runs"):
        category = "utility_power_sweep_selected_not_supported"
        action = "Power sweep found no selected > Stage-A random runs, including valid hash-noise control runs; stabilize probe protocol, then inspect selector/data learning signal."
    elif (memorization_like or easy_nll_tension) and anti_memorization_support["supports_selected"]:
        category = "anti_memorization_probe_supports_selector"
        action = "Selected loses to canonical multi-matched baseline but beats repeat-pressure-matched anti-memorization control; treat canonical strict failure as a Utility protocol confound candidate."
    elif memorization_like:
        category = "lm_train_memorization_proxy_gap"
        action = "Matched baseline learns train slices more easily because it is longer/repetition-heavier; inspect anti-memorization Utility controls before blaming Stage-B selection."
    elif train_gap < -0.001:
        category = "lm_train_signal_gap"
        action = "Feature-selected chunks are not producing comparable train-NLL learning; inspect text structure and probe train sampling."
    elif curation_margin["near_noise_floor"] or str(causal_mode) == "inconclusive_near_noise_floor":
        category = "utility_transfer_near_noise_floor"
        action = "Increase Utility probe budget/holdout power before changing selector policy."
    elif not curation_margin["positive"]:
        category = "eval_transfer_negative"
        action = "Feature gains do not transfer to held-out eval; inspect train/eval distribution alignment and baseline pairing."
    else:
        category = "strict_counterfactual_gap"
        action = "Feature gains transfer past random but not strict matched baseline; inspect matched baseline strength and OOD cells."
    framework_implication = _framework_implication(
        category=category,
        stage_c_pass=stage_c_pass,
        probe_valid=probe_valid,
        feature_stronger=feature_stronger,
        curation_margin=curation_margin,
        power=power,
        anti_memorization_support=anti_memorization_support,
        token_exposure_caveat=token_exposure_caveat,
    )
    return {
        "category": category,
        "action": action,
        "framework_implication": framework_implication,
        "stage_c_pass": bool(stage_c_pass),
        "feature_stronger_than_random": feature_stronger,
        "probe_valid": probe_valid,
        "causal_mode": causal_mode,
        "train_audit_gap": round(train_gap, 8),
        "curation_margin": curation_margin,
        "strict_margin": strict_margin,
        "matched_memorization_proxy_gap": memorization_gap,
        "learning_signal_diagnostic": learning_signal,
        "anti_memorization_diagnostic_baseline": anti_memorization_support,
        "power_sweep": power,
    }


def _dataset_report(
    dataset: str,
    meta: Dict[str, Any],
    selector_audit: Dict[str, Any],
    anti_memorization_reports: Dict[str, Any],
    power_sweep_report: Dict[str, Any],
    profile: str,
) -> Dict[str, Any]:
    aggregate = _utility_aggregate(meta)
    evidence = _utility_evidence(meta)
    stage_c = meta.get("stage_c_core_validation") or {}
    comparisons = _selector_comparisons(selector_audit, dataset)
    feature_random = _verdict_summary(comparisons, "stageA_random")
    feature_matched = _verdict_summary(comparisons, "multi_matched_stageA_random")
    transfer = _classify_transfer_gap(
        stage_c_pass=bool(stage_c.get("passed")),
        feature_random=feature_random,
        feature_matched=feature_matched,
        comparisons=comparisons,
        evidence=evidence,
        aggregate=aggregate,
        anti_memorization_probe=_anti_memorization_probe_result(anti_memorization_reports, dataset, profile),
        power_sweep=power_sweep_report,
    )
    return {
        "dataset": str(dataset),
        "feature_space": {
            "vs_stageA_random": feature_random,
            "vs_multi_matched_stageA_random": feature_matched,
        },
        "utility": {
            "evidence_tier": evidence.get("evidence_tier"),
            "failure_reason": evidence.get("failure_reason"),
            "small_lm_probe_gain_score": meta.get("small_lm_probe_gain_score"),
            "curation_benefit_status": evidence.get("curation_benefit_status") or aggregate.get("curation_benefit_status"),
            "strict_counterfactual_status": evidence.get("strict_counterfactual_status") or aggregate.get("strict_counterfactual_status"),
            "probe_sensitivity_status": evidence.get("probe_sensitivity_status") or aggregate.get("probe_sensitivity_status"),
            "causal_utility_audit": aggregate.get("causal_utility_audit") or evidence.get("causal_utility_audit"),
        },
        "transfer_gap": transfer,
    }


def build_report(run_summary: Dict[str, Any], profile: str) -> Dict[str, Any]:
    selector_audit = load_json(SELECTOR_BASELINE_AUDIT_PATH) if SELECTOR_BASELINE_AUDIT_PATH.exists() else {}
    anti_memorization_reports = _load_anti_memorization_reports()
    power_sweep_report = load_json(UTILITY_POWER_SWEEP_REPORT_PATH) if UTILITY_POWER_SWEEP_REPORT_PATH.exists() else {}
    profile_payload = _profile_payload(run_summary, profile)
    datasets = {
        str(dataset): _dataset_report(str(dataset), meta, selector_audit, anti_memorization_reports, power_sweep_report, profile)
        for dataset, meta in profile_payload.items()
        if not str(dataset).startswith("_") and isinstance(meta, dict)
    }
    categories: Dict[str, int] = {}
    for payload in datasets.values():
        category = str((payload.get("transfer_gap") or {}).get("category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    return {
        "schema_version": "utility-transfer-gap-report-v1",
        "profile": profile,
        "purpose": "Explain cases where Stage-B feature-space gains do not transfer to Stage-C small-LM Utility.",
        "inputs": {
            "run_summary": str(RUN_SUMMARY_PATH),
            "selector_baseline_audit": str(SELECTOR_BASELINE_AUDIT_PATH),
            "anti_memorization_probe_report": str(ANTI_MEMORIZATION_PROBE_REPORT_PATH),
            "anti_memorization_probe_reports": anti_memorization_reports.get("paths") or [],
            "utility_probe_power_sweep": str(UTILITY_POWER_SWEEP_REPORT_PATH),
        },
        "summary": {
            "dataset_count": int(len(datasets)),
            "transfer_gap_categories": categories,
        },
        "datasets": datasets,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Utility Transfer Gap Report",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Dataset count: `{(report.get('summary') or {}).get('dataset_count')}`",
        "",
        "| Dataset | Feature > Random | Probe | Train Gap | Repeat Gap | Length Gap | Curation Delta/MDE | Category | Framework implication | Action |",
        "|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        feature = ((payload.get("feature_space") or {}).get("vs_stageA_random") or {})
        utility = payload.get("utility") or {}
        probe = (utility.get("probe_sensitivity_status") or {}).get("status")
        gap = payload.get("transfer_gap") or {}
        curation_margin = gap.get("curation_margin") or {}
        memorization = gap.get("matched_memorization_proxy_gap") or {}
        implication = gap.get("framework_implication") or {}
        learning_gaps = ((gap.get("learning_signal_diagnostic") or {}).get("gaps_selected_minus_baseline") or {})
        anti = gap.get("anti_memorization_diagnostic_baseline") or {}
        power = gap.get("power_sweep") or {}
        lines.append(
            f"| {dataset} | {feature.get('verdict')} | {probe} | "
            f"{float(gap.get('train_audit_gap') or 0):+.8f} | "
            f"{float(memorization.get('repeat_pressure_delta') or 0):+.6f} | "
            f"{float(memorization.get('word_count_delta') or 0):+.3f} | "
            f"{float(curation_margin.get('delta_nll') or 0):+.8f}/{float(curation_margin.get('mde') or 0):.8f} | "
            f"{gap.get('category')} | {implication.get('status')} | {gap.get('action')} |"
        )
        if gap.get("category") == "lm_train_memorization_proxy_gap":
            lines.append(
                f"| {dataset} detail | selected-minus-baseline learning diagnostic | template gap "
                f"{float(learning_gaps.get('template_density') or 0):+.6f} | moderate difficulty gap "
                f"{float(learning_gaps.get('moderate_difficulty_share') or 0):+.6f} | "
                f"{float(learning_gaps.get('unique_bigram_ratio') or 0):+.6f} | | | learning_diagnostic | | |"
            )
        if anti.get("available"):
            lines.append(
                f"| {dataset} anti-mem | {anti.get('baseline')} | supports_selected={anti.get('supports_selected')} | "
                f"{float(anti.get('train_audit_gap') or 0):+.8f} | | | "
                f"{float(anti.get('delta_nll') or 0):+.8f}/{float(anti.get('minimum_detectable_delta_nll_95_max') or 0):.8f} | "
                f"{anti.get('causal_mode')} | anti_mem_diagnostic | effect_to_mde={anti.get('effect_to_mde_ratio_min')} |"
            )
        if power.get("available"):
            lines.append(
                f"| {dataset} power | compatible={power.get('compatible_runs')} valid={power.get('probe_valid_runs')} | "
                f"selected_gt_random={power.get('selected_gt_random_runs')} | "
                f"valid_selected_gt_random={power.get('valid_selected_gt_random_runs')} | | | "
                f"stable_probe_valid={power.get('stable_probe_valid')} | power_sweep | | {power.get('recommended_next_action')} |"
            )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Utility transfer-gap report.")
    parser.add_argument("--run-summary", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(load_json(args.run_summary), str(args.profile))
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[21] utility transfer gap json: {args.output}", flush=True)
    print(f"[21] utility transfer gap md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
