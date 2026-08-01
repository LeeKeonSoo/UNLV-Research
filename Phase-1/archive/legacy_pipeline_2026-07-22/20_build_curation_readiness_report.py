#!/usr/bin/env python3
"""Build a dataset-level curation readiness and failure-triage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, load_json, save_json


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "curation_readiness_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "curation_readiness_report.md"
SELECTOR_BASELINE_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"
UTILITY_TRANSFER_GAP_REPORT_PATH = OUTPUT_DIR / "validation" / "utility_transfer_gap_report.json"
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


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


def _probe_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = _utility_aggregate(meta)
    evidence = _utility_evidence(meta)
    return evidence.get("probe_sensitivity_status") or aggregate.get("probe_sensitivity_status") or {}


def _curation_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = _utility_aggregate(meta)
    evidence = _utility_evidence(meta)
    return evidence.get("curation_benefit_status") or aggregate.get("curation_benefit_status") or {}


def _strict_status(meta: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = _utility_aggregate(meta)
    evidence = _utility_evidence(meta)
    return evidence.get("strict_counterfactual_status") or aggregate.get("strict_counterfactual_status") or {}


def _top_items(values: Any, limit: int = 8) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values[:limit]]


def _profile_config_hint(profile: str) -> str:
    candidates = [
        CONFIG_DIR / f"{profile}.json",
        CONFIG_DIR / f"{profile}_probe.json",
        CONFIG_DIR / "core_proxy_length_recurrence_guard_probe.json",
        CONFIG_DIR / "curation_profiles.json",
    ]
    seen: set[Path] = set()
    for path in candidates + sorted(CONFIG_DIR.glob("*.json")):
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        profiles = payload.get("profiles") if isinstance(payload, dict) else None
        if isinstance(profiles, dict) and str(profile) in profiles:
            return str(path.relative_to(Path(__file__).resolve().parent))
    return "configs/curation_profiles.json"


def _replicated_power_sweep_presets(power_sweep: Dict[str, Any]) -> List[str]:
    family = str(power_sweep.get("best_replicated_valid_family") or "")
    replicates = (power_sweep.get("replicated_valid_family_replicates") or {}).get(family)
    if not family or not isinstance(replicates, list):
        return []
    return [f"{family}_b{int(replicate)}" for replicate in replicates]


def _recommended_action(
    *,
    profile: str,
    dataset: str,
    profile_config: str,
    stage_c: Dict[str, Any],
    probe: Dict[str, Any],
    curation: Dict[str, Any],
    strict: Dict[str, Any],
    evidence: Dict[str, Any],
    selector_feature_audit: Dict[str, Any],
    transfer_gap: Dict[str, Any],
) -> Dict[str, Any]:
    coverage_pass = bool(stage_c.get("coverage_pass"))
    probe_valid = bool(probe.get("destructive_probe_valid", probe.get("probe_valid")))
    probe_status = str(probe.get("status") or "")
    selected_beats_random = bool(curation.get("selected_beats_random"))
    selected_beats_matched = bool(strict.get("selected_beats_multi_matched"))
    strict_status = str(strict.get("status") or "")
    token_caveat = bool(probe.get("token_exposure_confounded") or probe.get("token_exposure_inconclusive"))
    failure_reason = str(evidence.get("failure_reason") or evidence.get("utility_failure_reason") or "")
    feature_vs_random = str((selector_feature_audit.get("stageA_random") or {}).get("verdict") or "")
    feature_vs_matched = str((selector_feature_audit.get("multi_matched_stageA_random") or {}).get("verdict") or "")
    feature_stronger_than_random = feature_vs_random == "selected_meaningfully_stronger"
    transfer_category = str(transfer_gap.get("category") or "")

    if bool(stage_c.get("passed")):
        if token_caveat:
            return {
                "category": "stage_c_ready_with_token_exposure_caveat",
                "priority": 2,
                "action": "Current Stage-C development validation passes, but token-exposure diagnostics are inconclusive; keep as targeted follow-up and rerun certification-grade Utility before promotion.",
                "command_hint": f"python 04_generate_subsets.py --profiles {profile_config} && python 14_run_utility_causal_diagnostics.py --profile {profile} --datasets {dataset} && python 20_build_curation_readiness_report.py --profile {profile}",
            }
        return {
            "category": "stage_c_ready",
            "priority": 2,
            "action": "Current Stage-C development validation passes; run certification-grade validation before promoting the profile globally.",
            "command_hint": f"python 14_run_utility_causal_diagnostics.py --profile {profile} --datasets {dataset} && python 20_build_curation_readiness_report.py --profile {profile}",
        }
    if transfer_category == "anti_memorization_probe_supports_selector":
        return {
            "category": "anti_memorization_probe_supports_selector",
            "priority": 1,
            "action": "Focused anti-memorization Utility control supports the selected subset; revise the reported strict-baseline controls before selector changes or certification claims.",
            "command_hint": f"python 25_build_stage_c_protocol_decision_report.py && python 26_build_strict_baseline_control_report.py",
            "transfer_gap_category": transfer_category,
            "additional_blocker": "coverage_not_ready" if not coverage_pass else None,
        }
    if not coverage_pass:
        return {
            "category": "coverage_not_ready",
            "priority": 1,
            "action": "Fix Coverage readiness before Utility tuning.",
            "command_hint": "python 04_generate_subsets.py && python validate_outputs.py",
        }
    if transfer_category == "probe_preset_candidate_available":
        power_sweep = transfer_gap.get("power_sweep") or {}
        power_sweep = power_sweep if isinstance(power_sweep, dict) else {}
        replicated_presets = _replicated_power_sweep_presets(power_sweep)
        best_replicated_family = str(power_sweep.get("best_replicated_valid_family") or "")
        if replicated_presets:
            return {
                "category": "probe_preset_standardization",
                "priority": 1,
                "action": f"Power sweep found replicated valid selected>Stage-A-random family `{best_replicated_family}`; standardize this Stage-C protocol candidate and rerun Stage-C before selector/Core tuning.",
                "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --presets {' '.join(replicated_presets)} --force",
                "transfer_gap_category": transfer_category,
                "replicated_stage_c_protocol_candidate": {
                    "family": best_replicated_family,
                    "presets": replicated_presets,
                },
            }
        return {
            "category": "probe_preset_standardization",
            "priority": 1,
            "action": "Power sweep found valid selected>Stage-A-random presets, but default Utility evidence is still unstable; standardize the stronger probe preset and rerun Stage-C before selector/Core tuning.",
            "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --presets stronger_probe_b0 --force",
            "transfer_gap_category": transfer_category,
        }
    if not probe_valid:
        return {
            "category": "probe_power_or_control_design",
            "priority": 1,
            "action": "Do not tune selector yet; increase Utility sensitivity power or inspect control-arm construction.",
            "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --force",
        }
    if transfer_category == "utility_transfer_near_noise_floor":
        return {
            "category": "utility_transfer_near_noise_floor",
            "priority": 1,
            "action": "Feature selection is stronger, but Utility transfer is below MDE; increase probe/holdout power before selector changes.",
            "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --force",
            "transfer_gap_category": transfer_category,
        }
    if transfer_category == "utility_power_sweep_selected_not_supported":
        return {
            "category": "utility_power_sweep_selected_not_supported",
            "priority": 1,
            "action": "Power sweep found no selected > Stage-A random runs; stabilize Utility protocol and inspect selector/data learning signal before claiming curated readiness.",
            "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets <dataset> --aggregate-only",
            "transfer_gap_category": transfer_category,
        }
    if transfer_category == "lm_train_signal_gap":
        return {
            "category": "lm_train_signal_gap",
            "priority": 1,
            "action": "Feature selection is stronger, but selected train slices produce weaker LM train-NLL learning; inspect text structure and train sampling.",
            "command_hint": f"python 21_build_utility_transfer_gap_report.py --profile {profile}",
            "transfer_gap_category": transfer_category,
        }
    if transfer_category == "lm_train_memorization_proxy_gap":
        return {
            "category": "lm_train_memorization_proxy_gap",
            "priority": 1,
            "action": "Selected chunks are stronger in feature space, but the matched baseline may train more easily because it is longer/repetition-heavier; add anti-memorization Utility controls before selector tuning.",
            "command_hint": f"python 21_build_utility_transfer_gap_report.py --profile {profile}",
            "transfer_gap_category": transfer_category,
        }
    if feature_stronger_than_random and (failure_reason == "selected_below_stageA_random" or not selected_beats_random):
        return {
            "category": "feature_space_utility_transfer_gap",
            "priority": 1,
            "action": "Selector improves feature-space quality/learnability versus Stage-A random, but LM Utility does not transfer; avoid blind selector tuning and inspect Utility eval/probe alignment.",
            "command_hint": f"python 15_run_selector_baseline_audit.py --datasets {dataset} && python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --force",
            "feature_vs_stageA_random": feature_vs_random,
            "feature_vs_multi_matched": feature_vs_matched,
        }
    if failure_reason == "selected_below_stageA_random" or not selected_beats_random:
        return {
            "category": "selector_underperforms_stageA_random",
            "priority": 1,
            "action": "Run selector ablation/baseline audits; selected subset is not beating feasible Stage-A random.",
            "command_hint": "python 15_run_selector_baseline_audit.py && python 17_run_policy_ablation_audit.py",
        }
    if token_caveat:
        return {
            "category": "utility_signal_with_token_exposure_caveat",
            "priority": 2,
            "action": "Keep selector changes conservative and confirm token-inventory stress with larger probe budget.",
            "command_hint": f"python 19_run_utility_probe_power_sweep.py --profile {profile} --datasets {dataset} --presets train_eval_hash_noise_b0 --force",
        }
    if not selected_beats_matched or strict_status in {"strict_negative", "matched_baseline_inconclusive"}:
        return {
            "category": "strict_counterfactual_not_ready",
            "priority": 2,
            "action": "Selected beats random or probe is valid, but strict multi-matched counterfactual evidence is not ready.",
            "command_hint": "python 15_run_selector_baseline_audit.py",
        }
    return {
        "category": "ready_or_near_ready",
        "priority": 3,
        "action": "Dataset is ready under current Stage-C evidence, or only certification protocol upgrades remain.",
        "command_hint": "python 13_run_paper_release.py --execute",
    }


def _dataset_report(dataset: str, meta: Dict[str, Any], *, profile: str) -> Dict[str, Any]:
    stage_c = meta.get("stage_c_core_validation") or {}
    evidence = _utility_evidence(meta)
    probe = _probe_status(meta)
    curation = _curation_status(meta)
    strict = _strict_status(meta)
    selector_feature_audit = _selector_feature_audit(dataset)
    transfer_gap = _transfer_gap(dataset)
    control_margins = probe.get("control_margins") or {}
    stage_c_pass = bool(stage_c.get("passed"))
    profile_config = _profile_config_hint(profile)
    recommended = _recommended_action(
        profile=profile,
        dataset=dataset,
        profile_config=profile_config,
        stage_c=stage_c,
        probe=probe,
        curation=curation,
        strict=strict,
        evidence=evidence,
        selector_feature_audit=selector_feature_audit,
        transfer_gap=transfer_gap,
    )
    framework_implication = transfer_gap.get("framework_implication") or {
        "status": "strict_counterfactual_not_resolved",
        "selector_policy_action": "hold",
        "strict_baseline_action": "inspect",
        "interpretation": "No Utility transfer-gap framework implication was available.",
    }
    if stage_c_pass:
        transfer_status = str((transfer_gap.get("framework_implication") or {}).get("status") or "")
        token_caveat = bool(
            transfer_status == "stage_c_development_ready_with_token_exposure_caveat"
            or probe.get("token_exposure_confounded")
            or probe.get("token_exposure_inconclusive")
        )
        framework_implication = {
            "status": (
                "stage_c_development_ready_with_token_exposure_caveat"
                if token_caveat
                else "stage_c_development_ready"
            ),
            "selector_policy_action": "hold",
            "strict_baseline_action": "certification_followup",
            "interpretation": (
                "The subset passes current Stage-C development validation, but token-inventory stress remains a caveat; keep Utility evidence as validation-only and rerun certification-grade Stage C before promotion."
                if token_caveat
                else "The subset passes current Stage-C development validation; keep Utility evidence as validation-only and run certification-grade follow-up before promotion."
            ),
            "previous_transfer_gap_status": transfer_status or None,
        }
    blockers = evidence.get("certification_blockers") or []
    protocol_blockers = evidence.get("protocol_blockers") or []
    signal_blockers = evidence.get("signal_blockers") or []
    return {
        "dataset": str(dataset),
        "selected_records": int(meta.get("selected_records") or 0),
        "processed_records": int(meta.get("processed_records") or meta.get("source_records") or 0),
        "selection_ratio": _safe_float(meta.get("selection_ratio")),
        "stage_c": {
            "passed": bool(stage_c.get("passed")),
            "coverage_pass": bool(stage_c.get("coverage_pass")),
            "utility_axis_pass": bool(stage_c.get("utility_axis_pass")),
            "final_utility_axis_pass": bool(stage_c.get("final_utility_axis_pass")),
            "final_certification_scope": stage_c.get("final_certification_scope"),
        },
        "coverage": {
            "score": _safe_float(meta.get("subset_coverage_retention_score")),
            "domain_support_pass": bool(stage_c.get("coverage_domain_support_pass")),
            "style_support_pass": bool(stage_c.get("coverage_style_support_pass")),
            "semantic_support_pass": bool(stage_c.get("coverage_semantic_support_pass")),
            "backbone_pass": bool(stage_c.get("coverage_backbone_pass")),
        },
        "utility": {
            "score": _safe_float(meta.get("small_lm_probe_gain_score")),
            "evidence_tier": evidence.get("evidence_tier"),
            "failure_reason": evidence.get("failure_reason"),
            "probe_status": probe.get("status"),
            "destructive_probe_valid": bool(probe.get("destructive_probe_valid", probe.get("probe_valid"))),
            "token_exposure_confounded": bool(probe.get("token_exposure_confounded")),
            "token_exposure_inconclusive": bool(probe.get("token_exposure_inconclusive")),
            "selected_beats_stageA_random": bool(curation.get("selected_beats_random")),
            "curation_status": curation.get("status"),
            "selected_beats_multi_matched": bool(strict.get("selected_beats_multi_matched")),
            "strict_status": strict.get("status"),
            "strict_min_gain": _safe_float(evidence.get("strict_min_gain")),
            "strict_min_delta_nll": _safe_float(evidence.get("strict_min_delta_nll")),
            "strict_min_delta_nll_ci_low": _safe_float(evidence.get("strict_min_delta_nll_ci_low")),
            "max_mde_95": _safe_float(evidence.get("max_minimum_detectable_delta_nll_95")),
            "control_margins": control_margins,
        },
        "selector_feature_audit": selector_feature_audit,
        "utility_transfer_gap": transfer_gap,
        "framework_implication": framework_implication,
        "recommended_next_action": recommended,
        "blockers": {
            "protocol_count": int(evidence.get("protocol_blocker_count") or len(protocol_blockers)),
            "signal_count": int(evidence.get("signal_blocker_count") or len(signal_blockers)),
            "top_protocol": _top_items(protocol_blockers),
            "top_signal": _top_items(signal_blockers),
            "top_certification": _top_items(blockers),
        },
    }


_SELECTOR_BASELINE_AUDIT_CACHE: Dict[str, Any] | None = None
_UTILITY_TRANSFER_GAP_CACHE: Dict[str, Any] | None = None


def _selector_baseline_audit() -> Dict[str, Any]:
    global _SELECTOR_BASELINE_AUDIT_CACHE
    if _SELECTOR_BASELINE_AUDIT_CACHE is not None:
        return _SELECTOR_BASELINE_AUDIT_CACHE
    if not SELECTOR_BASELINE_AUDIT_PATH.exists():
        _SELECTOR_BASELINE_AUDIT_CACHE = {}
        return {}
    try:
        payload = load_json(SELECTOR_BASELINE_AUDIT_PATH)
    except Exception:
        payload = {}
    _SELECTOR_BASELINE_AUDIT_CACHE = payload if isinstance(payload, dict) else {}
    return _SELECTOR_BASELINE_AUDIT_CACHE


def _selector_feature_audit(dataset: str) -> Dict[str, Any]:
    payload = ((_selector_baseline_audit().get("datasets") or {}).get(str(dataset)) or {})
    comparisons = payload.get("comparisons") if isinstance(payload, dict) else None
    if not isinstance(comparisons, dict):
        return {}
    out: Dict[str, Any] = {}
    for baseline_name, comparison in comparisons.items():
        verdict = (comparison or {}).get("verdict") or {}
        if not isinstance(verdict, dict):
            verdict = {}
        out[str(baseline_name)] = {
            "verdict": verdict.get("verdict"),
            "quality_delta": verdict.get("quality_delta"),
            "learnability_delta": verdict.get("learnability_delta"),
            "redundancy_risk_delta": verdict.get("redundancy_risk_delta"),
        }
    return out


def _utility_transfer_gap_report() -> Dict[str, Any]:
    global _UTILITY_TRANSFER_GAP_CACHE
    if _UTILITY_TRANSFER_GAP_CACHE is not None:
        return _UTILITY_TRANSFER_GAP_CACHE
    if not UTILITY_TRANSFER_GAP_REPORT_PATH.exists():
        _UTILITY_TRANSFER_GAP_CACHE = {}
        return {}
    try:
        payload = load_json(UTILITY_TRANSFER_GAP_REPORT_PATH)
    except Exception:
        payload = {}
    _UTILITY_TRANSFER_GAP_CACHE = payload if isinstance(payload, dict) else {}
    return _UTILITY_TRANSFER_GAP_CACHE


def _transfer_gap(dataset: str) -> Dict[str, Any]:
    payload = ((_utility_transfer_gap_report().get("datasets") or {}).get(str(dataset)) or {})
    gap = payload.get("transfer_gap") if isinstance(payload, dict) else None
    return gap if isinstance(gap, dict) else {}


def build_report(run_summary: Dict[str, Any], profile: str) -> Dict[str, Any]:
    profile_payload = _profile_payload(run_summary, profile)
    datasets = {
        str(dataset): _dataset_report(str(dataset), meta, profile=profile)
        for dataset, meta in profile_payload.items()
        if not str(dataset).startswith("_") and isinstance(meta, dict)
    }
    categories: Dict[str, int] = {}
    for payload in datasets.values():
        category = str((payload.get("recommended_next_action") or {}).get("category") or "unknown")
        categories[category] = categories.get(category, 0) + 1
    ready = [
        name
        for name, payload in datasets.items()
        if bool(((payload.get("stage_c") or {}).get("passed")))
    ]
    return {
        "schema_version": "curation-readiness-report-v1",
        "profile": profile,
        "purpose": "Dataset-level readiness and failure triage for the Core-Metric-Policy curation framework.",
        "policy": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "utility_scope": "Stage C only; never selector objective",
        },
        "summary": {
            "dataset_count": int(len(datasets)),
            "stage_c_ready_dataset_count": int(len(ready)),
            "not_ready_dataset_count": int(len(datasets) - len(ready)),
            "recommended_action_categories": categories,
        },
        "datasets": datasets,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    lines = [
        "# Curation Readiness Report",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Dataset count: `{summary.get('dataset_count')}`",
        f"- Stage-C ready datasets: `{summary.get('stage_c_ready_dataset_count')}`",
        f"- Not ready datasets: `{summary.get('not_ready_dataset_count')}`",
        "",
        "## Dataset Triage",
        "",
        "| Dataset | Stage C | Coverage | Probe | Token caveat | Feature > Random | Feature > Matched | Utility > Random | Utility > Matched | Evidence tier | Failure reason | Framework implication | Next action |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        stage_c = payload.get("stage_c") or {}
        coverage = payload.get("coverage") or {}
        utility = payload.get("utility") or {}
        action = payload.get("recommended_next_action") or {}
        implication = payload.get("framework_implication") or {}
        selector_audit = payload.get("selector_feature_audit") or {}
        feature_random = (selector_audit.get("stageA_random") or {}).get("verdict") or "-"
        feature_matched = (selector_audit.get("multi_matched_stageA_random") or {}).get("verdict") or "-"
        token_caveat = bool(utility.get("token_exposure_confounded") or utility.get("token_exposure_inconclusive"))
        lines.append(
            f"| {dataset} | {'pass' if stage_c.get('passed') else 'fail'} | "
            f"{'pass' if stage_c.get('coverage_pass') else 'fail'} ({float(coverage.get('score') or 0):.3f}) | "
            f"{utility.get('probe_status')} | {token_caveat} | "
            f"{feature_random} | {feature_matched} | "
            f"{utility.get('selected_beats_stageA_random')} | {utility.get('selected_beats_multi_matched')} | "
            f"{utility.get('evidence_tier')} | {utility.get('failure_reason')} | "
            f"{implication.get('status') or '-'} | {action.get('category')} |"
        )
    lines.extend(["", "## Recommended Commands", ""])
    for dataset, payload in (report.get("datasets") or {}).items():
        action = payload.get("recommended_next_action") or {}
        implication = payload.get("framework_implication") or {}
        selector_audit = payload.get("selector_feature_audit") or {}
        command = str(action.get("command_hint") or "").replace("<dataset>", str(dataset))
        lines.extend([
            f"### {dataset}",
            "",
            f"- Action: {action.get('action')}",
            f"- Feature-space vs Stage-A random: `{(selector_audit.get('stageA_random') or {}).get('verdict') or 'not_audited'}`",
            f"- Feature-space vs multi-matched: `{(selector_audit.get('multi_matched_stageA_random') or {}).get('verdict') or 'not_audited'}`",
            f"- Framework implication: `{implication.get('status') or 'not_available'}`",
            f"- Selector policy action: `{implication.get('selector_policy_action') or 'not_available'}`",
            f"- Strict baseline action: `{implication.get('strict_baseline_action') or 'not_available'}`",
            f"- Command hint: `{command}`",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curation readiness/failure-triage report.")
    parser.add_argument("--run-summary", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(load_json(args.run_summary), str(args.profile))
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[20] curation readiness json: {args.output}", flush=True)
    print(f"[20] curation readiness md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
