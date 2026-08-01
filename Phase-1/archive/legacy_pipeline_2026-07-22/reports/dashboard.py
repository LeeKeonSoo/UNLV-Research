#!/usr/bin/env python3
"""Generate a compact HTML dashboard for the generic data evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from data_eval_common import DASHBOARD_PATH, RUN_MANIFEST_PATH, SCHEMA_VERSION, SCORED_DIR, load_json


SCORING_MANIFEST_PATH = SCORED_DIR / "scoring_manifest.json"
UTILITY_SENSITIVITY_AUDIT_PATH = Path(__file__).resolve().parents[1] / "outputs" / "validation" / "utility_sensitivity_audit.json"


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_utility_sensitivity_audit() -> Dict[str, Any]:
    if not UTILITY_SENSITIVITY_AUDIT_PATH.exists():
        return {}
    try:
        payload = json.loads(UTILITY_SENSITIVITY_AUDIT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_dashboard_html(run_manifest: Dict[str, Any], scoring_manifest: Dict[str, Any]) -> str:
    profile_cards = []
    utility_sensitivity_audit = _load_utility_sensitivity_audit()
    utility_sensitivity_datasets = utility_sensitivity_audit.get("datasets") if isinstance(utility_sensitivity_audit, dict) else {}
    if not isinstance(utility_sensitivity_datasets, dict):
        utility_sensitivity_datasets = {}
    for profile_name, profile in run_manifest["profiles"].items():
        rows = []
        for dataset, meta in profile["datasets"].items():
            utility_details = meta.get("utility_probe_details") or {}
            utility_meta = utility_details.get("aggregate") or {}
            utility_evidence = utility_meta.get("utility_evidence_summary") or {}
            coverage_details = meta.get("coverage_details") or {}
            selector_diagnostics = meta.get("selector_diagnostics") or {}
            quality_band_balance = (
                ((selector_diagnostics.get("iterations") or [{}])[-1].get("quota_diagnostics") or {})
                .get("quality_band_distribution_balance")
                or {}
            )
            source_support = coverage_details.get("source_coverage_support") or {}
            domain_support = coverage_details.get("domain_coverage_support") or {}
            style_support = coverage_details.get("style_coverage_support") or {}
            semantic_support = coverage_details.get("semantic_coverage_support") or {}
            learning_signal_support = coverage_details.get("learning_signal_coverage_diagnostic") or {}
            learning_signal_gaps = learning_signal_support.get("gaps_selected_minus_baseline") or {}
            canonical_baseline = utility_meta.get("canonical_baseline") or "baseline_multi_matched_stageA_random"
            canonical_in_domain = (utility_details.get("in_domain") or {}).get(canonical_baseline) or {}
            utility_mode = utility_meta.get("mode") or (meta.get("stage_c_core_validation") or {}).get("utility_mode") or "single_eval"
            utility_mean_gain = utility_evidence.get("canonical_mean_gain", meta.get("small_lm_probe_gain_score", meta.get("fixed_token_probe_gain_score", 0.0)))
            strict_min_gain = utility_evidence.get("strict_min_gain", utility_meta.get("reported_small_lm_probe_gain_score_min"))
            strict_min_delta = utility_evidence.get("strict_min_delta_nll", utility_meta.get("min_delta_nll"))
            strict_min_ci_low = utility_evidence.get("strict_min_delta_nll_ci_low", utility_meta.get("min_delta_nll_ci_low"))
            mde_delta = utility_evidence.get("max_minimum_detectable_delta_nll_95", utility_meta.get("max_minimum_detectable_delta_nll_95"))
            effect_to_mde = utility_evidence.get("min_effect_to_mde_ratio", utility_meta.get("min_effect_to_mde_ratio"))
            detectable_fraction = utility_evidence.get("min_detectable_effect_fraction", utility_meta.get("min_detectable_effect_fraction"))
            min_rel_gain = utility_evidence.get("strict_min_relative_nll_gain", utility_meta.get("min_relative_nll_gain"))
            stress_min = utility_meta.get("stress_reported_small_lm_probe_gain_score_min")
            certification_shadow = utility_meta.get("certification_shadow") or {}
            cert_ready = utility_evidence.get("certification_ready", certification_shadow.get("certification_ready"))
            final_scope = utility_evidence.get("final_certification_scope") or utility_meta.get("final_certification_scope") or "-"
            final_scope_ready = utility_evidence.get(
                "final_scope_certification_ready",
                utility_meta.get("final_scope_certification_ready"),
            )
            in_domain_ready = utility_evidence.get(
                "in_domain_certification_ready",
                utility_meta.get("in_domain_certification_ready"),
            )
            cross_domain_ready = utility_evidence.get(
                "cross_domain_certification_ready",
                utility_meta.get("cross_domain_certification_ready"),
            )
            domain_specific_ready = utility_evidence.get(
                "domain_specific_certification_ready",
                utility_meta.get("domain_specific_certification_ready"),
            )
            general_purpose_ready = utility_evidence.get(
                "general_purpose_certification_ready",
                utility_meta.get("general_purpose_certification_ready"),
            )
            evidence_tier = utility_evidence.get("evidence_tier") or certification_shadow.get("evidence_tier") or "-"
            signal_status = utility_evidence.get("signal_status") or "-"
            sensitivity_payload = utility_sensitivity_datasets.get(str(dataset)) or {}
            sensitivity_order = sensitivity_payload.get("probe_sensitivity") or {}
            sensitivity_root = sensitivity_payload.get("root_cause_decision") or {}
            probe_valid = utility_evidence.get(
                "utility_probe_valid",
                utility_meta.get("utility_probe_valid", sensitivity_order.get("probe_valid", sensitivity_order.get("order_pass"))),
            )
            curation_status = utility_evidence.get("curation_benefit_status") or utility_meta.get("curation_benefit_status") or {}
            strict_counterfactual_status = (
                utility_evidence.get("strict_counterfactual_status")
                or utility_meta.get("strict_counterfactual_status")
                or {}
            )
            selected_gt_random = curation_status.get("selected_beats_random")
            if selected_gt_random is None:
                selected_gt_random = sensitivity_order.get("selected_gt_random")
            selected_gt_matched = strict_counterfactual_status.get("selected_beats_multi_matched")
            if selected_gt_matched is None:
                selected_gt_matched = utility_meta.get("utility_strict_pass")
            failure_analysis = utility_meta.get("utility_failure_analysis") or {}
            root_cause = (
                (utility_evidence.get("probe_sensitivity_status") or {}).get("root_cause")
                or sensitivity_root.get("primary_hypothesis")
                or failure_analysis.get("evidence_aware_failure_reason")
                or "-"
            )
            stability = (certification_shadow.get("stability_analysis") or {}).get("combined_effective") or {}
            worst_cells = certification_shadow.get("worst_cells") or {}
            worst_in = worst_cells.get("in_domain") or {}
            worst_ood = worst_cells.get("ood") or {}
            blocker_categories = certification_shadow.get("blocker_categories") or {}
            protocol_blockers = utility_evidence.get("protocol_blockers") or blocker_categories.get("protocol") or []
            signal_blockers = utility_evidence.get("signal_blockers") or blocker_categories.get("signal") or []
            blockers = utility_evidence.get("certification_blockers") or certification_shadow.get("blockers") or []
            causal_audit = utility_meta.get("causal_utility_audit") or failure_analysis.get("causal_utility_audit") or {}
            failure_mode = utility_evidence.get("failure_mode") or failure_analysis.get("failure_mode") or "-"
            causal_failure_mode = utility_evidence.get("causal_failure_mode") or causal_audit.get("dominant_failure_mode") or "-"
            causal_train_gap = causal_audit.get("mean_selected_minus_baseline_train_audit_delta_nll")
            ood_pair_count = utility_evidence.get("ood_pair_count", utility_meta.get("ood_pair_count"))
            if ood_pair_count is None:
                ood_pair_count = len(utility_meta.get("pairwise_ood_results") or {})
            blocker_text = ", ".join(str(x) for x in blockers[:3])
            if len(blockers) > 3:
                blocker_text += f", +{len(blockers) - 3} more"
            worst_text = "-"
            if worst_in or worst_ood:
                worst_bits = []
                if worst_in:
                    worst_bits.append(f"in={_fmt(utility_evidence.get('worst_in_domain_delta_nll', worst_in.get('delta_nll')), 6)}")
                if worst_ood:
                    worst_pair = utility_evidence.get("worst_ood_pair") or worst_ood.get("pair") or worst_ood.get("eval_dataset") or "ood"
                    worst_bits.append(f"{worst_pair}={_fmt(utility_evidence.get('worst_ood_delta_nll', worst_ood.get('delta_nll')), 6)}")
                worst_text = ", ".join(worst_bits)
            train_steps_text = "-"
            if canonical_in_domain:
                train_steps_text = (
                    f"{canonical_in_domain.get('selected_effective_train_steps_mean', '-')}/"
                    f"{canonical_in_domain.get('baseline_effective_train_steps_mean', '-')}"
                )
            exposure_text = "-"
            if canonical_in_domain:
                exposure_text = (
                    f"{_fmt(canonical_in_domain.get('selected_train_token_exposure_ratio_mean'), 2)}/"
                    f"{_fmt(canonical_in_domain.get('baseline_train_token_exposure_ratio_mean'), 2)}"
                    f" target={_fmt(canonical_in_domain.get('train_epochs_mean'), 1)}x"
                )
            rows.append(
                f"<tr><td>{dataset}</td><td>{meta['selected_records']}</td>"
                f"<td>{_fmt(meta['selection_ratio'])}</td>"
                f"<td>{_fmt(meta['subset_coverage_retention_score'])}</td>"
                f"<td>{domain_support.get('support_scope', '-')}</td>"
                f"<td>{_fmt(source_support.get('distribution_similarity'))}/{_fmt(source_support.get('retained_bucket_ratio'))}</td>"
                f"<td>{_fmt(domain_support.get('distribution_similarity'))}/{_fmt(domain_support.get('retained_bucket_ratio'))}</td>"
                f"<td>{_fmt(style_support.get('distribution_similarity'))}/{_fmt(style_support.get('retained_bucket_ratio'))}</td>"
                f"<td>{'pass' if semantic_support.get('cluster_backbone_pass') else 'fail'}</td>"
                f"<td>{_fmt(quality_band_balance.get('distribution_similarity_after'))}</td>"
                f"<td>{quality_band_balance.get('policy', '-')}</td>"
                f"<td>{_fmt(learning_signal_gaps.get('unique_bigram_ratio'), 4)}</td>"
                f"<td>{_fmt(learning_signal_gaps.get('concept_density'), 4)}</td>"
                f"<td>{_fmt(learning_signal_gaps.get('moderate_difficulty_share'), 4)}</td>"
                f"<td>{', '.join(str(x) for x in (learning_signal_support.get('risk_flags') or [])[:3]) or '-'}</td>"
                f"<td>{_fmt(utility_mean_gain, 6)}</td>"
                f"<td>{_fmt(strict_min_gain, 6)}</td>"
                f"<td>{_fmt(strict_min_delta, 6)}</td>"
                f"<td>{_fmt(strict_min_ci_low, 6)}</td>"
                f"<td>{_fmt(mde_delta, 6)}</td>"
                f"<td>{_fmt(effect_to_mde, 2)}</td>"
                f"<td>{_fmt(detectable_fraction, 2)}</td>"
                f"<td>{utility_mode}</td>"
                f"<td>{canonical_baseline}</td>"
                f"<td>{_fmt(min_rel_gain, 4) if min_rel_gain is not None else '-'}</td>"
                f"<td>{_fmt(stress_min, 4) if stress_min is not None else '-'}</td>"
                f"<td>{ood_pair_count}</td>"
                f"<td>{train_steps_text}</td>"
                f"<td>{exposure_text}</td>"
                f"<td>{'ready' if cert_ready else 'not ready'}</td>"
                f"<td>{final_scope}</td>"
                f"<td>{'ready' if final_scope_ready else 'not ready'}</td>"
                f"<td>{'ready' if in_domain_ready else 'not ready'}</td>"
                f"<td>{'ready' if cross_domain_ready else 'not ready'}</td>"
                f"<td>{'ready' if domain_specific_ready else 'not ready'}</td>"
                f"<td>{'ready' if general_purpose_ready else 'not ready'}</td>"
                f"<td>{'yes' if probe_valid is True else 'no' if probe_valid is False else '-'}</td>"
                f"<td>{'yes' if selected_gt_random is True else 'no' if selected_gt_random is False else '-'}</td>"
                f"<td>{'yes' if selected_gt_matched is True else 'no' if selected_gt_matched is False else '-'}</td>"
                f"<td>{evidence_tier}</td>"
                f"<td>{root_cause}</td>"
                f"<td>{signal_status}</td>"
                f"<td>{worst_text}</td>"
                f"<td>{len(protocol_blockers)}</td>"
                f"<td>{len(signal_blockers)}</td>"
                f"<td>{failure_mode}</td>"
                f"<td>{causal_failure_mode}</td>"
                f"<td>{_fmt(causal_train_gap, 6) if causal_train_gap is not None else '-'}</td>"
                f"<td>{'yes' if stability.get('noise_dominated') else 'no' if stability.get('noise_dominated') is not None else '-'}</td>"
                f"<td>{blocker_text or '-'}</td>"
                f"<td>{'pass' if (meta.get('stage_c_core_validation') or {}).get('passed') else 'fail'}</td></tr>"
            )
        profile_cards.append(
            f"""
            <section class="card">
              <h2>{profile_name}</h2>
              <p class="sub">Selection threshold: {_fmt(profile['selection_threshold'])}</p>
              <table>
                <thead><tr><th>Dataset</th><th>Selected</th><th>Ratio</th><th>Coverage</th><th>Domain Scope</th><th>Source Sim/Ret</th><th>Domain Sim/Ret</th><th>Style Sim/Ret</th><th>Semantic Backbone</th><th>Quality Band Sim</th><th>Quality Band Policy</th><th>Bigram Gap</th><th>Concept Gap</th><th>Difficulty Gap</th><th>Learning Risk Flags</th><th>Utility Mean</th><th>Strict Min</th><th>Min Delta</th><th>CI Low</th><th>MDE Delta</th><th>Effect/MDE</th><th>Detectable Frac</th><th>Utility Mode</th><th>Canonical Baseline</th><th>Min Rel Gain</th><th>Stress Min</th><th>OOD Pairs</th><th>Train Steps S/B</th><th>Exposure S/B</th><th>Utility Cert</th><th>Final Scope</th><th>Final Scope Cert</th><th>In-domain Cert</th><th>Cross-domain Cert</th><th>Domain-specific Cert</th><th>General-purpose Cert</th><th>Probe Valid</th><th>Selected &gt; Random</th><th>Selected &gt; Multi-matched</th><th>Evidence Tier</th><th>Root Cause</th><th>Signal Status</th><th>Worst Delta</th><th>Protocol Blockers</th><th>Signal Blockers</th><th>Failure Mode</th><th>Causal Mode</th><th>Train Gap</th><th>Noise Dominated</th><th>Cert Blockers</th><th>Stage C</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )

    metric_cards = []
    for dataset, meta in scoring_manifest["datasets"].items():
        core_rows = []
        for metric, stat in (meta.get("core_metrics") or {}).items():
            core_rows.append(
                f"<tr><td>{metric}</td><td>{_fmt(stat['mean'])}</td>"
                f"<td>{_fmt(stat['min'])}</td><td>{_fmt(stat['max'])}</td></tr>"
            )
        diag_rows = []
        for metric, stat in (meta.get("diagnostic_metrics") or {}).items():
            diag_rows.append(
                f"<tr><td>{metric}</td><td>{_fmt(stat['mean'])}</td>"
                f"<td>{_fmt(stat['min'])}</td><td>{_fmt(stat['max'])}</td></tr>"
            )
        sections = [
            "<h3>Core Selection Metrics</h3>"
            "<table><thead><tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead>"
            f"<tbody>{''.join(core_rows)}</tbody></table>"
        ]
        if diag_rows:
            sections.append(
                "<h3>Diagnostic Metrics</h3>"
                "<table><thead><tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead>"
                f"<tbody>{''.join(diag_rows)}</tbody></table>"
            )
        metric_cards.append(
            f"""
            <section class="card">
              <h2>{dataset}</h2>
              <p class="sub">Scored records: {meta['records']}</p>
              {''.join(sections)}
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Training Data Evaluation Dashboard</title>
  <style>
    :root {{
      --bg: #f3f5f7;
      --fg: #18212b;
      --muted: #5d6b79;
      --card: #ffffff;
      --line: #d7dde4;
      --accent: #215c98;
      --accent-soft: #dce8f7;
    }}
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: linear-gradient(180deg, #eef2f6 0%, #f9fbfc 100%); color: var(--fg); }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 48px 28px 80px; }}
    h1 {{ margin: 0 0 8px; font-size: 2.3rem; line-height: 1.1; }}
    p.lead {{ margin: 0 0 28px; max-width: 850px; color: var(--muted); line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 20px 22px; box-shadow: 0 10px 35px rgba(18, 33, 47, 0.06); }}
    .sub {{ color: var(--muted); margin-top: 0; }}
    .banner {{ background: linear-gradient(135deg, #15324b, #215c98); color: #fff; border-radius: 20px; padding: 26px 28px; margin-bottom: 22px; }}
    .badge {{ display: inline-block; background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; padding: 6px 11px; font-size: 0.8rem; margin-right: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); font-size: 0.94rem; }}
    th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; font-size: 0.78rem; }}
    td:first-child {{ font-weight: 600; }}
  </style>
</head>
<body>
  <main>
    <section class="banner">
      <span class="badge">{SCHEMA_VERSION}</span>
      <span class="badge">Threshold-controlled subsets</span>
      <h1>Training Data Evaluation Dashboard</h1>
      <p class="lead">This dashboard summarizes the canonical validated selection metrics, the retained diagnostic proxies, profile-driven subset generation, and subset-level coverage retention behavior for the current prepared-input datasets.</p>
    </section>
    <section>
      <h2>Metric Distributions</h2>
      <div class="grid">
        {''.join(metric_cards)}
      </div>
    </section>
    <section>
      <h2>Profile Outcomes</h2>
      <div class="grid">
        {''.join(profile_cards)}
      </div>
    </section>
  </main>
</body>
</html>"""


def build_dashboard(run_manifest_path: Path = RUN_MANIFEST_PATH, scoring_manifest_path: Path = SCORING_MANIFEST_PATH) -> Path:
    run_manifest = load_json(run_manifest_path)
    scoring_manifest = load_json(scoring_manifest_path)
    html = build_dashboard_html(run_manifest, scoring_manifest)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(html, encoding="utf-8")
    return DASHBOARD_PATH
