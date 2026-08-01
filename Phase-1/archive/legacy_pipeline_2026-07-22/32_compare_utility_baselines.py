#!/usr/bin/env python3
"""Compare Stage-C Utility baselines under one certification-grade protocol."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import (
    ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
    CANONICAL_UTILITY_BASELINE,
    INDEX_DB_PATH,
    OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
    SCORING_MANIFEST_PATH,
    _anti_memorization_bucket_from_scored_record,
    _fingerprint_uids,
    _matched_bucket_baseline_pool,
    _multi_matched_stagea_baseline_pool,
    _nuisance_matched_bucket_from_scored_record,
    _passes_gates,
    _score_with_probe_buckets,
    _stage_a_gate,
    _utility_probe_config,
)


DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "configs" / "style_taxonomy_alignment_probe.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validation"
BASELINE_ORDER = (
    CANONICAL_UTILITY_BASELINE,
    OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
    ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
)


def _progress(message: str) -> None:
    print(f"[32] {message}", flush=True)


def _profile_payload(path: Path, profile_name: str) -> Dict[str, Any]:
    profile = ((load_json(path).get("profiles") or {}).get(profile_name))
    if not isinstance(profile, dict):
        raise RuntimeError(f"Profile {profile_name!r} not found in {path}")
    return profile


def _compact_pool_diagnostics(payload: Dict[str, Any], limit: int = 200) -> Dict[str, Any]:
    out = dict(payload)
    for key in ("bucket_targets", "bucket_available", "bucket_selected", "bucket_available_exact"):
        value = out.get(key)
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda item: (-int(item[1]), str(item[0])))
            out[key] = {str(k): int(v) for k, v in items[:limit]}
            out[f"{key}_total_buckets"] = int(len(value))
            out[f"{key}_truncated"] = bool(len(value) > limit)
    return out


def _pool_fidelity(name: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    available = diagnostics.get("bucket_available")
    if not isinstance(available, dict):
        available = diagnostics.get("bucket_available_exact")
    available = available if isinstance(available, dict) else {}
    available_count = sum(1 for value in available.values() if int(value) > 0)
    bucket_count = int(diagnostics.get("bucket_count") or len(available))
    return {
        "baseline": name,
        "matched_pool_count": int(diagnostics.get("matched_pool_count") or 0),
        "bucket_count": bucket_count,
        "buckets_with_exact_control": int(available_count),
        "exact_bucket_availability_ratio": round(available_count / max(bucket_count, 1), 6),
        "exclude_selected": bool(diagnostics.get("exclude_selected")),
        "excluded_selected_records": int(diagnostics.get("excluded_selected_records") or 0),
        "matching_policy": diagnostics.get("matching_policy") or "exact_bucket_match",
        "matched_variables": diagnostics.get("matched_variables"),
        "excluded_selector_target_variables": diagnostics.get("excluded_selector_target_variables"),
        "fallback_order": diagnostics.get("fallback_order"),
    }


def _build_pools(
    *,
    stage_a_records: list[Dict[str, Any]],
    selected_records: list[Dict[str, Any]],
    seed: int,
    pool_multiplier: int,
) -> Dict[str, Dict[str, Any]]:
    pools: Dict[str, Dict[str, Any]] = {}
    canonical_uids, canonical_diag = _multi_matched_stagea_baseline_pool(
        baseline_records=stage_a_records,
        selected_records=selected_records,
        seed=seed,
        pool_multiplier=pool_multiplier,
        exclude_selected=True,
    )
    pools[CANONICAL_UTILITY_BASELINE] = {"allowed_uids": canonical_uids, "diagnostics": canonical_diag}

    for name, bucket_fn, contract in (
        (
            OPERATIONAL_COUNTERFACTUAL_CANDIDATE_BASELINE,
            _nuisance_matched_bucket_from_scored_record,
            {
                "matching_policy": "exact_length_style_domain_repeat_pressure",
                "matched_variables": ["length", "style", "domain", "repeat_pressure"],
                "excluded_selector_target_variables": ["quality", "redundancy_risk"],
                "fallback_order": [],
            },
        ),
        (
            ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
            _anti_memorization_bucket_from_scored_record,
            {
                "matching_policy": "exact_quality_length_style_domain_repeat_pressure",
                "matched_variables": ["quality", "length", "style", "domain", "repeat_pressure"],
                "excluded_selector_target_variables": [],
                "fallback_order": [],
            },
        ),
    ):
        uids, diagnostics = _matched_bucket_baseline_pool(
            baseline_records=stage_a_records,
            selected_records=selected_records,
            bucket_fn=bucket_fn,
            seed=seed,
            pool_multiplier=pool_multiplier,
            exclude_selected=True,
        )
        diagnostics.update(contract)
        pools[name] = {"allowed_uids": uids, "diagnostics": diagnostics}
    return pools


def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    stability = result.get("stability_diagnostics") or {}
    return {
        "small_lm_probe_gain_score": result.get("small_lm_probe_gain_score"),
        "delta_nll": result.get("delta_nll"),
        "delta_nll_min": result.get("delta_nll_min"),
        "delta_nll_ci_low": result.get("delta_nll_ci_low"),
        "minimum_detectable_delta_nll_95_max": result.get("minimum_detectable_delta_nll_95_max"),
        "effect_to_mde_ratio_min": result.get("effect_to_mde_ratio_min"),
        "detectable_effect_fraction": result.get("detectable_effect_fraction"),
        "positive_run_fraction": stability.get("positive_run_fraction"),
        "ci_positive_fraction": stability.get("ci_positive_fraction"),
        "run_count": len(result.get("per_bucket_runs") or []),
    }


def _run_dataset(dataset: str, profile_name: str, profiles_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    profile = _profile_payload(profiles_path, profile_name)
    stage_a = _stage_a_gate(profile)
    probe_cfg = _utility_probe_config(profile, evaluation_mode="certification")
    run_summary = load_json(RUN_SUMMARY_PATH)
    dataset_summary = (((run_summary.get("profiles") or {}).get(profile_name) or {}).get(dataset) or {})
    selected_path = Path(
        str(
            dataset_summary.get("output_path")
            or (OUTPUT_DIR / "subsets" / profile_name / f"{dataset}.jsonl")
        )
    )
    scored_path = Path(str((((load_json(SCORING_MANIFEST_PATH).get("datasets") or {}).get(dataset) or {}).get("path") or "")))
    if not selected_path.exists() or not scored_path.exists():
        raise FileNotFoundError(f"Missing selected/scored input for {profile_name}:{dataset}")

    selected_records = list(iter_jsonl_records_resilient(selected_path))
    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    _progress(f"{dataset}: selected={len(selected_records)}; loading disjoint Stage-A pool")
    stage_a_records = [
        record
        for record in iter_jsonl_records_resilient(scored_path)
        if str(record.get("chunk_uid") or "") not in selected_uids and _passes_gates(record, stage_a)
    ]
    seed = int(probe_cfg.get("sampling_hash_seed") or 42)
    multiplier = int(((profile.get("selector") or {}).get("matched_baseline_pool_multiplier") or 4))
    pools = _build_pools(
        stage_a_records=stage_a_records,
        selected_records=selected_records,
        seed=seed,
        pool_multiplier=multiplier,
    )
    _progress(
        f"{dataset}: pools ready "
        + " ".join(f"{name}={len(pools[name]['allowed_uids'])}" for name in BASELINE_ORDER)
    )

    text_map = {str(record.get("chunk_uid") or ""): str(record.get("text") or "") for record in selected_records}
    context_cache: Dict[Any, Any] = {}
    selected_sequence_cache: Dict[Any, Any] = {}
    results: Dict[str, Any] = {}
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    try:
        for name in BASELINE_ORDER:
            _progress(f"{dataset}: certification comparison start baseline={name}")
            allowed_uids = pools[name]["allowed_uids"]
            results[name] = _score_with_probe_buckets(
                conn,
                context_cache=context_cache,
                selected_sequence_cache=selected_sequence_cache,
                selected_records=selected_records,
                text_map=text_map,
                baseline_variant=name,
                baseline_allowed_uids=allowed_uids,
                baseline_uid_fingerprint=_fingerprint_uids(allowed_uids),
                train_dataset=dataset,
                eval_dataset=dataset,
                probe_cfg=probe_cfg,
                eval_token_budget=int(probe_cfg["eval_token_budget"]),
                holdout_buckets=list(probe_cfg.get("holdout_buckets") or []),
                progress_label=f"baseline-comparison:{dataset}:{name}",
            )
            _progress(f"{dataset}: baseline={name} delta_nll={results[name].get('delta_nll')}")
    finally:
        conn.close()

    summaries = {name: _result_summary(results[name]) for name in BASELINE_ORDER}
    return {
        "schema_version": "utility-baseline-comparison-v1",
        "profile": profile_name,
        "dataset": dataset,
        "purpose": "Compare canonical, operational-candidate, and anti-memorization Stage-C baselines under one certification protocol.",
        "selector_objective_scope": "Stage C validation only; never selector objective",
        "canonical_baseline": CANONICAL_UTILITY_BASELINE,
        "promotion_policy": "No baseline promotion from a single dataset or single run; require replicated certification evidence across positive and raw-like cases.",
        "probe_protocol": {
            key: probe_cfg.get(key)
            for key in (
                "model_name",
                "train_token_budget",
                "eval_token_budget",
                "bootstrap_samples",
                "max_train_steps",
                "train_epochs",
                "train_audit_token_budget",
                "min_probe_bucket_count",
                "holdout_buckets",
                "seeds",
            )
        },
        "selected_records": len(selected_records),
        "stage_a_control_records": len(stage_a_records),
        "pool_fidelity": {name: _pool_fidelity(name, pools[name]["diagnostics"]) for name in BASELINE_ORDER},
        "pool_diagnostics": {name: _compact_pool_diagnostics(pools[name]["diagnostics"]) for name in BASELINE_ORDER},
        "result_summaries": summaries,
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Certification Utility Baseline Comparison",
        "",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Profile: `{report.get('profile')}`",
        f"- Canonical baseline: `{report.get('canonical_baseline')}`",
        "",
        "| Baseline | Delta NLL | Min delta | CI low | Effect/MDE min | Positive runs | Pool records | Exact bucket availability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in BASELINE_ORDER:
        result = (report.get("result_summaries") or {}).get(name) or {}
        fidelity = (report.get("pool_fidelity") or {}).get(name) or {}
        lines.append(
            f"| {name} | {result.get('delta_nll')} | {result.get('delta_nll_min')} | "
            f"{result.get('delta_nll_ci_low')} | {result.get('effect_to_mde_ratio_min')} | "
            f"{result.get('positive_run_fraction')} | {fidelity.get('matched_pool_count')} | "
            f"{fidelity.get('exact_bucket_availability_ratio')} |"
        )
    lines.extend(["", f"- Promotion policy: {report.get('promotion_policy')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Utility baselines under certification-grade settings.")
    parser.add_argument("--datasets", nargs="+", default=["openwebtext2_subset"])
    parser.add_argument("--profile", default="style_taxonomy_alignment_probe")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        report = _run_dataset(str(dataset), str(args.profile), args.profiles)
        json_path = args.output_dir / f"utility_baseline_comparison_{dataset}.json"
        md_path = args.output_dir / f"utility_baseline_comparison_{dataset}.md"
        save_json(json_path, report)
        _write_markdown(report, md_path)
        _progress(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
