#!/usr/bin/env python3
"""Build a Stage-C protocol decision report for curation readiness triage."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_protocol_decision_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "stage_c_protocol_decision_report.md"
CURATION_READINESS_REPORT_PATH = OUTPUT_DIR / "validation" / "curation_readiness_report.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
UTILITY_POWER_SWEEP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_probe_power_sweep.json"
ANTI_MEMORIZATION_PROBE_REPORT_PATH = OUTPUT_DIR / "validation" / "anti_memorization_probe_report.json"
ANTI_MEMORIZATION_PROBE_REPORT_GLOB = "anti_memorization_probe_report*.json"
REPLICATE_PRESET_RE = re.compile(r"^(?P<family>.+)_b(?P<replicate>\d+)$")


def _load_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_anti_memorization_reports(path: Path) -> Dict[str, Any]:
    reports = []
    paths = []
    validation_dir = OUTPUT_DIR / "validation"
    candidate_paths = []
    if path.exists():
        candidate_paths.append(path)
    candidate_paths.extend(sorted(validation_dir.glob(ANTI_MEMORIZATION_PROBE_REPORT_GLOB)))
    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = _load_optional(candidate)
        if payload.get("schema_version") != "anti-memorization-probe-report-v1":
            continue
        reports.append(payload)
        paths.append(str(candidate))
    return {"reports": reports, "paths": paths}


def _anti_memorization_for_dataset(report_bundle: Dict[str, Any], dataset: str, profile: str | None) -> Dict[str, Any]:
    reports = report_bundle.get("reports")
    if not isinstance(reports, list):
        reports = [report_bundle]
    for report in reports:
        if (
            isinstance(report, dict)
            and str(report.get("dataset") or "") == str(dataset)
            and (profile is None or str(report.get("profile") or "") == str(profile))
        ):
            return report
    return {}


def _best_power_preset(power_sweep: Dict[str, Any], dataset: str) -> str | None:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    decision = payload.get("decision") if isinstance(payload, dict) else None
    if isinstance(decision, dict):
        preset = decision.get("best_valid_selected_gt_random_preset")
        return str(preset) if preset else None
    return None


def _power_decision(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    decision = payload.get("decision") if isinstance(payload, dict) else None
    return decision if isinstance(decision, dict) else {}


def _power_sweep_for_profile(power_sweep: Dict[str, Any], profile: str | None) -> Dict[str, Any]:
    if profile is None:
        return power_sweep
    return power_sweep if str(power_sweep.get("profile") or "") == str(profile) else {}


def _valid_selected_presets(power_sweep: Dict[str, Any], dataset: str) -> list[str]:
    decision = _power_decision(power_sweep, dataset)
    presets = decision.get("valid_selected_gt_random_presets")
    if not isinstance(presets, list):
        return []
    return [str(preset) for preset in presets]


def _replicate_status_by_family(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, Dict[int, bool]]:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return {}
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


def _replicated_valid_family_replicates(power_sweep: Dict[str, Any], dataset: str) -> Dict[str, list[int]]:
    status_by_family = _replicate_status_by_family(power_sweep, dataset)
    return {
        family: sorted(status_by_replicate)
        for family, status_by_replicate in sorted(status_by_family.items())
        if len(status_by_replicate) >= 2 and all(status_by_replicate.values())
    }


def _replicated_valid_families(power_sweep: Dict[str, Any], dataset: str) -> list[str]:
    return sorted(_replicated_valid_family_replicates(power_sweep, dataset).keys())


def _best_replicated_preset(power_sweep: Dict[str, Any], dataset: str) -> str | None:
    payload = ((power_sweep.get("datasets") or {}).get(str(dataset)) or {})
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, dict):
        return None
    replicated = _replicated_valid_family_replicates(power_sweep, dataset)
    candidates = [
        f"{family}_b{replicate}"
        for family, replicates in replicated.items()
        for replicate in replicates
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda name: (
            float((runs.get(name) or {}).get("selected_minus_random") or 0.0),
            str(name),
        ),
    )


def _best_replicated_family(power_sweep: Dict[str, Any], dataset: str) -> str | None:
    preset = _best_replicated_preset(power_sweep, dataset)
    match = REPLICATE_PRESET_RE.match(str(preset or ""))
    return match.group("family") if match else None


def _global_power_sweep_decision(power_sweep: Dict[str, Any], datasets: Dict[str, Any]) -> Dict[str, Any]:
    dataset_names = sorted(str(dataset) for dataset in datasets)
    per_dataset = {dataset: _valid_selected_presets(power_sweep, dataset) for dataset in dataset_names}
    replicated_by_dataset = {
        dataset: _replicated_valid_families(power_sweep, dataset)
        for dataset in dataset_names
    }
    replicated_replicates_by_dataset = {
        dataset: _replicated_valid_family_replicates(power_sweep, dataset)
        for dataset in dataset_names
    }
    preset_sets = [set(presets) for presets in per_dataset.values() if presets]
    common = sorted(set.intersection(*preset_sets)) if len(preset_sets) == len(dataset_names) and preset_sets else []
    replicated_sets = [set(families) for families in replicated_by_dataset.values() if families]
    common_replicated = (
        sorted(set.intersection(*replicated_sets))
        if len(replicated_sets) == len(dataset_names) and replicated_sets
        else []
    )
    best_by_dataset = {
        dataset: _best_power_preset(power_sweep, dataset)
        for dataset in dataset_names
    }
    best_replicated_by_dataset = {
        dataset: _best_replicated_preset(power_sweep, dataset)
        for dataset in dataset_names
    }
    return {
        "dataset_count": len(dataset_names),
        "datasets_with_valid_selected_gt_random_preset": sum(1 for presets in per_dataset.values() if presets),
        "valid_selected_gt_random_presets_by_dataset": per_dataset,
        "replicated_valid_families_by_dataset": replicated_by_dataset,
        "replicated_valid_family_replicates_by_dataset": replicated_replicates_by_dataset,
        "best_valid_selected_gt_random_preset_by_dataset": best_by_dataset,
        "best_replicated_valid_preset_by_dataset": best_replicated_by_dataset,
        "common_valid_selected_gt_random_presets": common,
        "common_replicated_valid_families": common_replicated,
        "global_default_preset_available": bool(common),
        "global_replicated_default_family_available": bool(common_replicated),
        "recommendation": (
            "A replicated common Stage-C Utility family exists across all datasets; test it as a global certification candidate."
            if common_replicated
            else "No replicated common valid selected>Stage-A-random preset family exists across all datasets; do not promote a global Utility preset from the current sweep."
        ),
    }


def _anti_memorization_support(result: Dict[str, Any]) -> Dict[str, Any]:
    if not result:
        return {
            "available": False,
            "supports_selected": False,
            "delta_nll": None,
            "delta_nll_ci_low": None,
            "minimum_detectable_delta_nll_95_max": None,
        }
    delta = float(result.get("delta_nll") or 0.0)
    ci_low = float(result.get("delta_nll_ci_low") or 0.0)
    mde = float(result.get("minimum_detectable_delta_nll_95_max") or 0.0)
    return {
        "available": True,
        "supports_selected": bool(delta > 0.0 and ci_low > 0.0 and (mde <= 0.0 or abs(delta) > mde)),
        "delta_nll": round(delta, 8),
        "delta_nll_ci_low": round(ci_low, 8),
        "minimum_detectable_delta_nll_95_max": round(abs(mde), 8),
    }


def _protocol_decision(
    *,
    dataset: str,
    profile: str | None,
    readiness: Dict[str, Any],
    transfer_gap: Dict[str, Any],
    power_sweep: Dict[str, Any],
    anti_memorization: Dict[str, Any],
) -> Dict[str, Any]:
    stage_c = readiness.get("stage_c") or {}
    utility = readiness.get("utility") or {}
    action = readiness.get("recommended_next_action") or {}
    implication = readiness.get("framework_implication") or {}
    transfer = transfer_gap.get("transfer_gap") or {}
    category = str(action.get("category") or transfer.get("category") or "")
    best_preset = _best_power_preset(power_sweep, dataset)
    best_replicated_preset = _best_replicated_preset(power_sweep, dataset)
    best_replicated_family = _best_replicated_family(power_sweep, dataset)
    power_decision = _power_decision(power_sweep, dataset)
    anti_report = _anti_memorization_for_dataset(anti_memorization, dataset, profile)
    anti_result = anti_report.get("utility_result") or {}
    anti_support = _anti_memorization_support(anti_result)
    replicated_families = _replicated_valid_families(power_sweep, dataset)
    coverage_passed = bool(stage_c.get("coverage_pass"))
    selected_beats_random = bool(utility.get("selected_beats_stageA_random"))
    selected_beats_matched = bool(utility.get("selected_beats_multi_matched"))
    token_exposure_caveat = bool(utility.get("token_exposure_confounded") or utility.get("token_exposure_inconclusive"))
    operational_total_effect_pass = bool(coverage_passed and selected_beats_random)

    if category == "probe_preset_standardization" or transfer.get("category") == "probe_preset_candidate_available":
        status = "probe_protocol_candidate_not_certified"
        if best_replicated_family:
            decision = (
                "Power sweep contains a replicated valid selected>Stage-A-random "
                f"family `{best_replicated_family}`, but the default Stage-C probe "
                "remains unstable. Treat the family as a Stage-C protocol candidate only."
            )
            next_protocol_step = (
                f"Standardize replicated family `{best_replicated_family}` "
                f"(best preset `{best_replicated_preset}`) for this dataset before "
                "making any strict curation-readiness claim."
            )
        else:
            decision = (
                "Power sweep contains at least one valid selected>Stage-A-random preset, "
                "but the default Stage-C probe remains unstable. Treat the preset as a "
                "Stage-C protocol candidate only."
            )
            next_protocol_step = (
                f"Rerun and standardize `{best_preset}` for this dataset before making "
                "any operational curation-readiness claim."
                if best_preset
                else "Rerun the power sweep with the current profile and choose a stable probe preset."
            )
    elif operational_total_effect_pass and replicated_families and not token_exposure_caveat:
        status = "operational_total_effect_certification_candidate"
        decision = (
            "The selected subset beats a disjoint equal-budget Stage-A random "
            "control and has replicated selected>random Utility support. Matched "
            "controls remain conditional diagnostics rather than primary gates."
        )
        next_protocol_step = "Use as a certification candidate only after downstream safety, contamination, and forgetting checks are complete."
    elif operational_total_effect_pass:
        status = "operational_total_effect_development_ready"
        decision = (
            "The selected subset beats a disjoint equal-budget Stage-A random "
            "control, but certification still needs replicated protocol support "
            "and caveat handling."
        )
        next_protocol_step = "Rerun official Stage-C Utility with replicated certification settings and keep matched-control caveats explicit."
    elif anti_support["supports_selected"] and not selected_beats_random:
        status = "conditional_matched_support_without_total_effect"
        decision = (
            "Conditional matched diagnostics support the selected subset, but the "
            "primary total-effect comparison against Stage-A random does not. "
            "Treat this as mechanism evidence, not as a training-use claim."
        )
        next_protocol_step = "Keep the dataset rejected or abstained for training use; inspect retained/rejected slices before changing Stage B."
    else:
        status = "no_operational_utility_gain"
        decision = "Current Stage-C evidence is not sufficient for a curation-readiness claim."
        next_protocol_step = str(action.get("action") or "Inspect Stage-C evidence and rerun the relevant diagnostic.")

    return {
        "dataset": dataset,
        "protocol_status": status,
        "decision": decision,
        "next_protocol_step": next_protocol_step,
        "stage_c_passed": bool(stage_c.get("passed")),
        "operational_total_effect_pass": operational_total_effect_pass,
        "coverage_passed": coverage_passed,
        "probe_status": utility.get("probe_status"),
        "token_exposure_caveat": token_exposure_caveat,
        "primary_utility_estimand": "selected_vs_equal_budget_disjoint_stageA_random",
        "matched_controls_role": "conditional_mechanism_diagnostics_not_primary_gate",
        "utility_selected_beats_stageA_random": selected_beats_random,
        "utility_selected_beats_multi_matched": selected_beats_matched,
        "framework_implication": implication.get("status"),
        "recommended_action_category": action.get("category"),
        "best_valid_power_sweep_preset": best_preset,
        "best_replicated_power_sweep_preset": best_replicated_preset,
        "best_replicated_power_sweep_family": best_replicated_family,
        "valid_selected_gt_random_power_sweep_presets": _valid_selected_presets(power_sweep, dataset),
        "replicated_valid_power_sweep_families": replicated_families,
        "replicated_valid_power_sweep_family_replicates": _replicated_valid_family_replicates(power_sweep, dataset),
        "power_sweep_stable_probe_valid": bool(power_decision.get("stable_probe_valid")),
        "anti_memorization_diagnostic_available": anti_support["available"],
        "anti_memorization_supports_selected": anti_support["supports_selected"],
        "anti_memorization_delta_nll": anti_support["delta_nll"],
        "selector_policy_action": "hold",
        "utility_scope": "Stage C validation only; never selector objective",
    }


def build_report(
    readiness_report: Dict[str, Any],
    transfer_gap_report: Dict[str, Any],
    power_sweep_report: Dict[str, Any],
    anti_memorization_reports: Dict[str, Any],
) -> Dict[str, Any]:
    profile = readiness_report.get("profile") or transfer_gap_report.get("profile")
    matching_power_sweep = _power_sweep_for_profile(
        power_sweep_report,
        str(profile) if profile is not None else None,
    )
    datasets: Dict[str, Any] = {}
    for dataset, readiness in (readiness_report.get("datasets") or {}).items():
        transfer = ((transfer_gap_report.get("datasets") or {}).get(str(dataset)) or {})
        datasets[str(dataset)] = _protocol_decision(
            dataset=str(dataset),
            profile=str(profile) if profile is not None else None,
            readiness=readiness if isinstance(readiness, dict) else {},
            transfer_gap=transfer if isinstance(transfer, dict) else {},
            power_sweep=matching_power_sweep,
            anti_memorization=anti_memorization_reports,
        )
    certified = [
        dataset
        for dataset, payload in datasets.items()
        if payload.get("protocol_status") == "operational_total_effect_certification_candidate"
    ]
    global_power = _global_power_sweep_decision(matching_power_sweep, datasets)
    return {
        "schema_version": "stage-c-protocol-decision-report-v1",
        "profile": profile,
        "power_sweep_profile": power_sweep_report.get("profile"),
        "power_sweep_profile_matches": bool(matching_power_sweep),
        "purpose": "Record Stage-C protocol decisions without changing Stage-B selector objectives.",
        "framework_contract": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
            "sensitivity_baseline_policy": "common Stage-A baseline disjoint from all sensitivity arms",
            "primary_utility_estimand": "selected_vs_equal_budget_disjoint_stageA_random",
            "matched_controls_role": "conditional mechanism diagnostics, not primary certification gates",
        },
        "global_decision": {
            "profile_promoted": False,
            "certified_ready_dataset_count": len(certified),
            "global_default_utility_preset_available": global_power["global_default_preset_available"],
            "global_replicated_default_utility_family_available": global_power["global_replicated_default_family_available"],
            "common_valid_selected_gt_random_presets": global_power["common_valid_selected_gt_random_presets"],
            "common_replicated_valid_families": global_power["common_replicated_valid_families"],
            "interpretation": (
                "The framework is separating selected-subset construction from Stage-C validation. "
                "Current evidence supports targeted follow-up, not global promotion."
            ),
        },
        "global_power_sweep_decision": global_power,
        "datasets": datasets,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    global_decision = report.get("global_decision") or {}
    lines = [
        "# Stage-C Protocol Decision Report",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Profile promoted: `{global_decision.get('profile_promoted')}`",
        f"- Certified ready datasets: `{global_decision.get('certified_ready_dataset_count')}`",
        f"- Global default Utility preset available: `{global_decision.get('global_default_utility_preset_available')}`",
        f"- Global replicated Utility family available: `{global_decision.get('global_replicated_default_utility_family_available')}`",
        f"- Common valid presets: `{global_decision.get('common_valid_selected_gt_random_presets') or []}`",
        f"- Common replicated families: `{global_decision.get('common_replicated_valid_families') or []}`",
        "- Utility scope: `Stage C validation only; never selector objective`",
        "",
        "## Dataset Decisions",
        "",
        "| Dataset | Protocol status | Probe | Token caveat | Utility > Random | Utility > Matched | Operational pass | Best preset | Selector action |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.append(
            f"| {dataset} | {payload.get('protocol_status')} | {payload.get('probe_status')} | "
            f"{payload.get('token_exposure_caveat')} | {payload.get('utility_selected_beats_stageA_random')} | "
            f"{payload.get('utility_selected_beats_multi_matched')} | {payload.get('operational_total_effect_pass')} | "
            f"{payload.get('best_valid_power_sweep_preset') or '-'} | "
            f"{payload.get('selector_policy_action')} |"
        )
    lines.extend(["", "## Next Protocol Steps", ""])
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.extend([
            f"### {dataset}",
            "",
            f"- Decision: {payload.get('decision')}",
            f"- Next step: {payload.get('next_protocol_step')}",
            f"- Framework implication: `{payload.get('framework_implication')}`",
            f"- Valid selected>random presets: `{payload.get('valid_selected_gt_random_power_sweep_presets') or []}`",
            f"- Replicated valid families: `{payload.get('replicated_valid_power_sweep_families') or []}`",
            f"- Anti-memorization supports selected: `{payload.get('anti_memorization_supports_selected')}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-C protocol decision report.")
    parser.add_argument("--readiness-report", type=Path, default=CURATION_READINESS_REPORT_PATH)
    parser.add_argument("--transfer-gap-report", type=Path, default=UTILITY_TRANSFER_GAP_REPORT_PATH)
    parser.add_argument("--power-sweep-report", type=Path, default=UTILITY_POWER_SWEEP_REPORT_PATH)
    parser.add_argument("--anti-memorization-report", type=Path, default=ANTI_MEMORIZATION_PROBE_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        _load_optional(args.readiness_report),
        _load_optional(args.transfer_gap_report),
        _load_optional(args.power_sweep_report),
        _load_anti_memorization_reports(args.anti_memorization_report),
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[25] Stage-C protocol decision json: {args.output}", flush=True)
    print(f"[25] Stage-C protocol decision md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
