#!/usr/bin/env python3
"""Build the final LM-training curation decision report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, load_json, save_json


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "curation_decision_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "curation_decision_report.md"
CURATION_READINESS_REPORT_PATH = OUTPUT_DIR / "validation" / "curation_readiness_report.json"
STAGE_C_PROTOCOL_DECISION_REPORT_PATH = OUTPUT_DIR / "validation" / "stage_c_protocol_decision_report.json"
STRICT_BASELINE_CONTROL_REPORT_PATH = OUTPUT_DIR / "validation" / "strict_baseline_control_report.json"
SELECTOR_BASELINE_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"


def _load_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _profile_payload(run_summary: Dict[str, Any], profile: str | None) -> Dict[str, Any]:
    profiles = run_summary.get("profiles") or {}
    if profile and profile in profiles and isinstance(profiles[profile], dict):
        return profiles[profile]
    names = [
        str(name)
        for name, payload in profiles.items()
        if not str(name).startswith("_") and isinstance(payload, dict)
    ]
    if len(names) == 1:
        return profiles[names[0]]
    return {}


def _status(ok: bool, caveat: bool = False, missing: bool = False) -> str:
    if missing:
        return "missing"
    if ok and caveat:
        return "pass_with_caveat"
    return "pass" if ok else "fail"


def _verdict(payload: Dict[str, Any], baseline: str) -> Dict[str, Any]:
    return (((payload.get("comparisons") or {}).get(str(baseline)) or {}).get("verdict") or {})


def _evidence_item(
    *,
    stage: str,
    claim: str,
    status: str,
    evidence: Dict[str, Any],
    interpretation: str,
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "claim": claim,
        "status": status,
        "evidence": evidence,
        "interpretation": interpretation,
    }


def _stage_a_evidence(dataset: str, meta: Dict[str, Any], selector_audit: Dict[str, Any]) -> Dict[str, Any]:
    audit = ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
    selected_records = int(meta.get("selected_records") or audit.get("selected_records") or 0)
    processed_records = int(meta.get("processed_records") or meta.get("source_records") or 0)
    stage_a_records = int(audit.get("stage_a_records") or 0)
    candidate_excluding_selected = int(audit.get("stage_a_candidate_records_excluding_selected") or 0)
    hard_gate_ok = bool(stage_a_records > 0 and selected_records > 0 and stage_a_records >= selected_records)
    has_baseline_pool = bool(candidate_excluding_selected > 0)
    survival_rate = float(stage_a_records) / float(processed_records) if processed_records > 0 and stage_a_records else None
    return _evidence_item(
        stage="A",
        claim="chunk-level hard gate creates a usable Stage-A pool",
        status=_status(hard_gate_ok and has_baseline_pool),
        evidence={
            "processed_records": processed_records,
            "stage_a_records": stage_a_records,
            "selected_records": selected_records,
            "stage_a_candidate_records_excluding_selected": candidate_excluding_selected,
            "stage_a_survival_rate": round(survival_rate, 6) if survival_rate is not None else None,
        },
        interpretation=(
            "Stage A has a non-empty usable pool and leaves enough disjoint candidates for random/matched baselines."
            if hard_gate_ok and has_baseline_pool
            else "Stage A evidence is not sufficient to support downstream curation decisions."
        ),
    )


def _stage_b_evidence(dataset: str, selector_audit: Dict[str, Any]) -> List[Dict[str, Any]]:
    audit = ((selector_audit.get("datasets") or {}).get(str(dataset)) or {})
    random_verdict = _verdict(audit, "stageA_random")
    matched_verdict = _verdict(audit, "multi_matched_stageA_random")
    random_ok = str(random_verdict.get("verdict") or "") == "selected_meaningfully_stronger"
    matched_ok = str(matched_verdict.get("verdict") or "") == "selected_meaningfully_stronger"
    return [
        _evidence_item(
            stage="B",
            claim="selected subset is stronger than Stage-A random in Core feature space",
            status=_status(random_ok, missing=not bool(random_verdict)),
            evidence={
                "verdict": random_verdict.get("verdict"),
                "quality_delta": random_verdict.get("quality_delta"),
                "learnability_delta": random_verdict.get("learnability_delta"),
                "redundancy_risk_delta": random_verdict.get("redundancy_risk_delta"),
                "word_count_delta": random_verdict.get("word_count_delta"),
            },
            interpretation=(
                "Stage B improves the selected subset over feasible usable random data without using Utility as an objective."
                if random_ok
                else "Stage B does not yet show a clear Core-feature gain over Stage-A random."
            ),
        ),
        _evidence_item(
            stage="B",
            claim="selected subset remains stronger under multi-matched Stage-A comparison",
            status=_status(matched_ok, missing=not bool(matched_verdict)),
            evidence={
                "verdict": matched_verdict.get("verdict"),
                "quality_delta": matched_verdict.get("quality_delta"),
                "learnability_delta": matched_verdict.get("learnability_delta"),
                "redundancy_risk_delta": matched_verdict.get("redundancy_risk_delta"),
                "word_count_delta": matched_verdict.get("word_count_delta"),
            },
            interpretation=(
                "Core feature gains survive stricter matching, supporting the selector as a curation policy."
                if matched_ok
                else "Core feature gains do not survive the strict matched comparison."
            ),
        ),
    ]


def _stage_c_evidence(
    readiness: Dict[str, Any],
    protocol: Dict[str, Any],
    strict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    stage_c = readiness.get("stage_c") or {}
    utility = readiness.get("utility") or {}
    coverage = readiness.get("coverage") or {}
    protocol_status = str(protocol.get("protocol_status") or "")
    replicated_families = protocol.get("replicated_valid_power_sweep_families")
    if not isinstance(replicated_families, list):
        replicated_families = []
    token_caveat = bool(strict.get("token_exposure_caveat") or protocol.get("token_exposure_caveat"))
    anti_available = bool(strict.get("anti_memorization_diagnostic_available"))
    anti_supports = bool(strict.get("anti_memorization_supports_selected"))
    cert_claim = bool(strict.get("certification_claim_allowed"))
    operational_pass = bool(strict.get("operational_total_effect_pass") or protocol.get("operational_total_effect_pass"))
    return [
        _evidence_item(
            stage="C",
            claim="subset preserves required coverage",
            status=_status(bool(stage_c.get("coverage_pass"))),
            evidence={
                "coverage_pass": stage_c.get("coverage_pass"),
                "coverage_score": coverage.get("score"),
                "domain_support_pass": coverage.get("domain_support_pass"),
                "style_support_pass": coverage.get("style_support_pass"),
                "semantic_support_pass": coverage.get("semantic_support_pass"),
            },
            interpretation=(
                "The selected subset preserves the configured coverage constraints."
                if bool(stage_c.get("coverage_pass"))
                else "The selected subset does not preserve the configured coverage constraints."
            ),
        ),
        _evidence_item(
            stage="C",
            claim="selected subset shows primary total curation benefit",
            status=_status(bool(utility.get("selected_beats_stageA_random")) and bool(stage_c.get("coverage_pass"))),
            evidence={
                "primary_utility_estimand": "selected_vs_equal_budget_disjoint_stageA_random",
                "selected_beats_stageA_random": utility.get("selected_beats_stageA_random"),
                "operational_total_effect_pass": operational_pass,
                "curation_status": utility.get("curation_status"),
                "probe_status": utility.get("probe_status"),
            },
            interpretation=(
                "The selected subset beats feasible Stage-A random under the current Utility evidence."
                if bool(utility.get("selected_beats_stageA_random"))
                else "The selected subset does not beat feasible Stage-A random, so matched-control positives are mechanism evidence only."
            ),
        ),
        _evidence_item(
            stage="C",
            claim="conditional matched controls support mechanism analysis",
            status=_status(bool(utility.get("selected_beats_multi_matched") or anti_supports), caveat=True),
            evidence={
                "selected_beats_multi_matched": utility.get("selected_beats_multi_matched"),
                "strict_status": utility.get("strict_status"),
                "strict_min_delta_nll": utility.get("strict_min_delta_nll"),
                "strict_min_delta_nll_ci_low": utility.get("strict_min_delta_nll_ci_low"),
                "anti_memorization_supports_selected": anti_supports,
                "matched_controls_role": strict.get("matched_controls_role")
                or "conditional_mechanism_diagnostics_not_primary_gate",
            },
            interpretation=(
                "Matched controls support a conditional mechanism claim, not the total curation effect by themselves."
                if bool(utility.get("selected_beats_multi_matched") or anti_supports)
                else "Matched controls do not currently add supporting mechanism evidence."
            ),
        ),
        _evidence_item(
            stage="C",
            claim="token-exposure diagnostics are clean enough for certification",
            status=_status(not token_caveat, caveat=token_caveat),
            evidence={
                "token_exposure_caveat": token_caveat,
                "probe_status": utility.get("probe_status"),
            },
            interpretation=(
                "Token-exposure diagnostics do not add a caveat."
                if not token_caveat
                else "Token-exposure diagnostics remain a caveat; do not make a clean certification claim."
            ),
        ),
        _evidence_item(
            stage="C",
            claim="reported diagnostic controls address easy-NLL or memorization confounds",
            status=_status(anti_supports, missing=not anti_available),
            evidence={
                "anti_memorization_diagnostic_available": anti_available,
                "anti_memorization_supports_selected": anti_supports,
                "anti_memorization_evidence": strict.get("anti_memorization_evidence"),
            },
            interpretation=(
                "The anti-memorization diagnostic supports the selected subset."
                if anti_supports
                else "No supporting anti-memorization diagnostic evidence is available for this dataset."
            ),
        ),
        _evidence_item(
            stage="C",
            claim="Stage-C Utility protocol is stable enough for a certification claim",
            status=_status(bool(replicated_families) and cert_claim),
            evidence={
                "protocol_status": protocol_status,
                "replicated_valid_power_sweep_families": replicated_families,
                "certification_claim_allowed": cert_claim,
            },
            interpretation=(
                "The dataset has replicated primary Stage-C support and reported controls allow a certification claim."
                if bool(replicated_families) and cert_claim
                else "The dataset is not yet certification-ready under the current Stage-C protocol."
            ),
        ),
    ]


def _decision_from_evidence(
    readiness: Dict[str, Any],
    protocol: Dict[str, Any],
    strict: Dict[str, Any],
    stage_a_evidence: Dict[str, Any],
) -> Dict[str, Any]:
    stage_c = readiness.get("stage_c") or {}
    framework = readiness.get("framework_implication") or {}
    utility = readiness.get("utility") or {}
    stage_a = stage_a_evidence.get("evidence") or {}
    strict_status = str(strict.get("status") or "")
    protocol_status = str(protocol.get("protocol_status") or "")
    token_caveat = bool(strict.get("token_exposure_caveat") or protocol.get("token_exposure_caveat"))
    certification_claim_allowed = bool(strict.get("certification_claim_allowed"))
    selected_beats_random = bool(utility.get("selected_beats_stageA_random") or strict.get("primary_operational_selected_beats_stageA_random"))
    coverage_passed = bool(stage_c.get("coverage_pass") or strict.get("coverage_passed"))
    operational_total_effect_pass = bool(strict.get("operational_total_effect_pass") or (coverage_passed and selected_beats_random))
    stage_a_records = int(stage_a.get("stage_a_records") or 0)
    selected_records = int(stage_a.get("selected_records") or 0)
    baseline_candidates = int(stage_a.get("stage_a_candidate_records_excluding_selected") or 0)
    usable_data_sufficient = bool(
        stage_a_records > 0
        and selected_records > 0
        and stage_a_records >= selected_records
        and baseline_candidates > 0
    )

    caveats: List[str] = []
    if not usable_data_sufficient:
        caveats.append("insufficient_usable_data")
    if token_caveat:
        caveats.append("token_exposure_caveat")
    if protocol_status == "probe_protocol_candidate_not_certified":
        caveats.append("utility_probe_unstable")
    if not selected_beats_random and bool(strict.get("anti_memorization_supports_selected")):
        caveats.append("conditional_matched_support_without_total_effect")
    if not operational_total_effect_pass:
        caveats.append("no_operational_utility_gain")
    if not certification_claim_allowed and operational_total_effect_pass:
        caveats.append("needs_certification_utility")

    if not usable_data_sufficient:
        decision = "insufficient_usable_data"
        training_use = "do_not_train_insufficient_usable_data"
        operational_action = "insufficient_usable_data"
        certification_claim_allowed = False
        rationale = (
            "The Stage-A usable pool is insufficient to create both a non-empty selected subset "
            "and a disjoint validation baseline; the framework abstains from a training-use claim."
        )
    elif certification_claim_allowed:
        decision = "accepted_for_training"
        training_use = "certification_candidate"
        operational_action = "accept"
        rationale = "Primary Stage-C Utility evidence supports using this selected subset for training; matched controls are reported as diagnostics."
    elif token_caveat and operational_total_effect_pass:
        decision = "needs_certification_utility"
        training_use = "development_only_with_token_exposure_caveat"
        operational_action = "manual_review"
        rationale = "The subset is development-ready, but token-exposure diagnostics block a clean training certification claim."
    elif protocol_status == "probe_protocol_candidate_not_certified":
        decision = "utility_probe_unstable"
        training_use = "hold_for_stage_c_protocol_standardization"
        operational_action = "manual_review"
        rationale = "Stage B evidence is useful, but the Stage-C Utility protocol is not stable enough for a training-use claim."
    elif strict_status == "conditional_matched_support_without_total_effect":
        decision = "rejected_for_training"
        training_use = "do_not_use_without_followup"
        operational_action = "manual_review"
        rationale = "Matched controls support a conditional mechanism effect, but the subset does not beat the primary Stage-A-random Utility baseline."
    elif operational_total_effect_pass:
        decision = "accepted_for_training_with_caveat"
        training_use = "development_only"
        operational_action = "accept_with_caveat"
        rationale = "The subset shows primary operational curation benefit but lacks certification-grade replicated evidence."
    else:
        decision = "rejected_for_training"
        training_use = "do_not_use_without_followup"
        operational_action = "reject"
        rationale = "Current evidence does not support using the selected subset for the intended training claim."

    return {
        "decision": decision,
        "training_use": training_use,
        "operational_action": operational_action,
        "certification_claim_allowed": certification_claim_allowed,
        "usable_data_sufficient": usable_data_sufficient,
        "caveats": sorted(set(caveats)),
        "framework_implication": framework.get("status"),
        "rationale": rationale,
    }


def _dataset_decision(
    dataset: str,
    meta: Dict[str, Any],
    selector_audit: Dict[str, Any],
    readiness: Dict[str, Any],
    protocol: Dict[str, Any],
    strict: Dict[str, Any],
) -> Dict[str, Any]:
    stage_a_evidence = _stage_a_evidence(dataset, meta, selector_audit)
    evidence_matrix = [
        stage_a_evidence,
        *_stage_b_evidence(dataset, selector_audit),
        *_stage_c_evidence(readiness, protocol, strict),
    ]
    decision = _decision_from_evidence(readiness, protocol, strict, stage_a_evidence)
    next_step = (
        "Restore or collect more usable candidate data before Stage-B selection and Stage-C validation."
        if decision["decision"] == "insufficient_usable_data"
        else strict.get("next_step")
        or protocol.get("next_protocol_step")
        or (readiness.get("recommended_next_action") or {}).get("action")
    )
    return {
        "dataset": dataset,
        "decision": decision["decision"],
        "training_use": decision["training_use"],
        "operational_action": decision["operational_action"],
        "certification_claim_allowed": decision["certification_claim_allowed"],
        "usable_data_sufficient": decision["usable_data_sufficient"],
        "caveats": decision["caveats"],
        "framework_implication": decision["framework_implication"],
        "rationale": decision["rationale"],
        "selected_records": int(meta.get("selected_records") or readiness.get("selected_records") or 0),
        "selection_ratio": meta.get("selection_ratio", readiness.get("selection_ratio")),
        "evidence_matrix": evidence_matrix,
        "next_step": next_step,
        "utility_scope": "Stage C validation only; never selector objective",
    }


def build_report(
    run_summary: Dict[str, Any],
    readiness_report: Dict[str, Any],
    protocol_report: Dict[str, Any],
    strict_report: Dict[str, Any],
    selector_audit: Dict[str, Any],
) -> Dict[str, Any]:
    profile = (
        readiness_report.get("profile")
        or protocol_report.get("profile")
        or strict_report.get("profile")
    )
    profile_payload = _profile_payload(run_summary, str(profile) if profile else None)
    readiness_datasets = readiness_report.get("datasets") or {}
    protocol_datasets = protocol_report.get("datasets") or {}
    strict_datasets = strict_report.get("datasets") or {}
    dataset_names = sorted(set(readiness_datasets) | set(protocol_datasets) | set(strict_datasets))
    datasets = {
        dataset: _dataset_decision(
            dataset,
            (profile_payload.get(dataset) or {}) if isinstance(profile_payload, dict) else {},
            selector_audit,
            (readiness_datasets.get(dataset) or {}) if isinstance(readiness_datasets, dict) else {},
            (protocol_datasets.get(dataset) or {}) if isinstance(protocol_datasets, dict) else {},
            (strict_datasets.get(dataset) or {}) if isinstance(strict_datasets, dict) else {},
        )
        for dataset in dataset_names
    }
    decision_counts: Dict[str, int] = {}
    caveat_counts: Dict[str, int] = {}
    for payload in datasets.values():
        decision = str(payload.get("decision") or "unknown")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        for caveat in payload.get("caveats") or []:
            caveat_counts[str(caveat)] = caveat_counts.get(str(caveat), 0) + 1
    return {
        "schema_version": "curation-decision-report-v2",
        "profile": profile,
        "purpose": "Map Stage A/B/C evidence into explicit LM-training curation decisions.",
        "research_framing": {
            "input": "arbitrary candidate corpus",
            "output": "curated LM-training dataset or explicit abstention",
            "data_collection_scope": "upstream",
            "utility_scope": "Stage C validation only; never selector objective",
        },
        "framework_contract": {
            "stage_a": "chunk-level hard gate",
            "stage_b": "chunk-level selection",
            "stage_c": "subset-level validation",
            "decision_layer": "training-use claim over the selected subset",
        },
        "decision_policy": {
            "accepted_for_training": "certification-grade primary Stage-A-random Utility evidence supports training use",
            "accepted_for_training_with_caveat": "development Stage C passes but caveats must be reported",
            "needs_certification_utility": "Stage A/B evidence is good, but Stage-C Utility is not certification-grade",
            "utility_probe_unstable": "probe controls or preset stability block a training-use claim",
            "conditional_matched_support_without_total_effect": "matched controls support a mechanism claim but not the primary total curation effect",
            "rejected_for_training": "current evidence does not support training use",
            "insufficient_usable_data": "the usable pool cannot support selection plus disjoint validation evidence",
        },
        "summary": {
            "dataset_count": len(datasets),
            "decision_counts": decision_counts,
            "caveat_counts": caveat_counts,
            "certification_claim_allowed_dataset_count": sum(
                1 for payload in datasets.values() if bool(payload.get("certification_claim_allowed"))
            ),
        },
        "datasets": datasets,
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    lines = [
        "# Curation Decision Report",
        "",
        f"- Profile: `{report.get('profile')}`",
        "- Framing: `arbitrary candidate corpus -> curated LM-training dataset`",
        "- Utility scope: `Stage C validation only; never selector objective`",
        f"- Dataset count: `{summary.get('dataset_count')}`",
        f"- Certification-claim datasets: `{summary.get('certification_claim_allowed_dataset_count')}`",
        "",
        "## Dataset Decisions",
        "",
        "| Dataset | Decision | Operational action | Training use | Cert claim | Caveats | Next step |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.append(
            f"| {dataset} | {payload.get('decision')} | {payload.get('operational_action')} | {payload.get('training_use')} | "
            f"{payload.get('certification_claim_allowed')} | {', '.join(payload.get('caveats') or []) or 'none'} | "
            f"{payload.get('next_step')} |"
        )
    lines.extend(["", "## Evidence Matrix", ""])
    for dataset, payload in (report.get("datasets") or {}).items():
        lines.extend([
            f"### {dataset}",
            "",
            f"- Decision: `{payload.get('decision')}`",
            f"- Rationale: {payload.get('rationale')}",
            "",
            "| Stage | Claim | Status | Interpretation |",
            "|---|---|---|---|",
        ])
        for item in payload.get("evidence_matrix") or []:
            lines.append(
                f"| {item.get('stage')} | {item.get('claim')} | {item.get('status')} | {item.get('interpretation')} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final curation decision report.")
    parser.add_argument("--run-summary", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--readiness-report", type=Path, default=CURATION_READINESS_REPORT_PATH)
    parser.add_argument("--protocol-report", type=Path, default=STAGE_C_PROTOCOL_DECISION_REPORT_PATH)
    parser.add_argument("--strict-report", type=Path, default=STRICT_BASELINE_CONTROL_REPORT_PATH)
    parser.add_argument("--selector-audit", type=Path, default=SELECTOR_BASELINE_AUDIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        _load_optional(args.run_summary),
        _load_optional(args.readiness_report),
        _load_optional(args.protocol_report),
        _load_optional(args.strict_report),
        _load_optional(args.selector_audit),
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[27] curation decision json: {args.output}", flush=True)
    print(f"[27] curation decision md: {args.md_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
