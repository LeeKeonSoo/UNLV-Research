#!/usr/bin/env python3
"""Decompose Stage-C Utility matching variables under one certification protocol."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict

from data_eval_common import OUTPUT_DIR, RUN_SUMMARY_PATH, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import (
    INDEX_DB_PATH,
    SCORING_MANIFEST_PATH,
    _domain_bucket_from_scored_record,
    _fingerprint_uids,
    _length_bucket_from_scored_record,
    _matched_bucket_baseline_pool,
    _passes_gates,
    _quality_band_from_scored_record,
    _repeat_pressure_bucket_from_scored_record,
    _score_with_probe_buckets,
    _stage_a_gate,
    _style_bucket_from_scored_record,
    _utility_probe_config,
)


DEFAULT_PROFILES_PATH = Path(__file__).resolve().parent / "configs" / "style_taxonomy_alignment_probe.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validation"
STAGE_A_RANDOM = "baseline_stageA_random"
DECOMPOSITION_ARMS = (
    ("exact_length_style_domain", ("length", "style", "domain")),
    ("exact_length_style_domain_repeat", ("length", "style", "domain", "repeat_pressure")),
    ("exact_length_style_domain_repeat_quality", ("length", "style", "domain", "repeat_pressure", "quality")),
    (
        "exact_length_style_domain_repeat_quality_redundancy",
        ("length", "style", "domain", "repeat_pressure", "quality", "redundancy_risk"),
    ),
)


def _progress(message: str) -> None:
    print(f"[33] {message}", flush=True)


def _profile_payload(path: Path, profile_name: str) -> Dict[str, Any]:
    profile = ((load_json(path).get("profiles") or {}).get(profile_name))
    if not isinstance(profile, dict):
        raise RuntimeError(f"Profile {profile_name!r} not found in {path}")
    return profile


def _redundancy_risk_bucket(record: Dict[str, Any]) -> str:
    payload = ((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {})
    try:
        score = float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    if score < 0.10:
        return "redundancy_lt_0_10"
    if score < 0.20:
        return "redundancy_0_10_0_20"
    if score < 0.35:
        return "redundancy_0_20_0_35"
    if score < 0.50:
        return "redundancy_0_35_0_50"
    if score < 0.70:
        return "redundancy_0_50_0_70"
    return "redundancy_ge_0_70"


COMPONENTS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "length": _length_bucket_from_scored_record,
    "style": _style_bucket_from_scored_record,
    "domain": _domain_bucket_from_scored_record,
    "repeat_pressure": _repeat_pressure_bucket_from_scored_record,
    "quality": _quality_band_from_scored_record,
    "redundancy_risk": _redundancy_risk_bucket,
}


def _bucket_fn(variables: tuple[str, ...]) -> Callable[[Dict[str, Any]], str]:
    def bucket(record: Dict[str, Any]) -> str:
        return "|".join(f"{name}={COMPONENTS[name](record)}" for name in variables)

    return bucket


def _pool_fidelity(name: str, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "baseline": name,
        "matched_pool_count": int(diagnostics.get("matched_pool_count") or 0),
        "bucket_count": int(diagnostics.get("bucket_count") or 0),
        "matched_bucket_ratio": diagnostics.get("matched_bucket_ratio"),
        "matched_selected_reference_ratio": diagnostics.get("matched_selected_reference_ratio"),
        "exclude_selected": bool(diagnostics.get("exclude_selected")),
        "matched_variables": diagnostics.get("matched_variables"),
        "fallback_order": diagnostics.get("fallback_order"),
    }


def _result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    stability = result.get("stability_diagnostics") or {}
    return {
        "delta_nll": result.get("delta_nll"),
        "delta_nll_min": result.get("delta_nll_min"),
        "delta_nll_ci_low": result.get("delta_nll_ci_low"),
        "minimum_detectable_delta_nll_95_max": result.get("minimum_detectable_delta_nll_95_max"),
        "effect_to_mde_ratio_min": result.get("effect_to_mde_ratio_min"),
        "detectable_effect_fraction": result.get("detectable_effect_fraction"),
        "positive_run_fraction": stability.get("positive_run_fraction"),
        "run_count": len(result.get("per_bucket_runs") or []),
    }


def _build_pools(
    stage_a_records: list[Dict[str, Any]],
    selected_records: list[Dict[str, Any]],
    *,
    seed: int,
    pool_multiplier: int,
) -> Dict[str, Dict[str, Any]]:
    stage_a_uids = {str(record.get("chunk_uid") or "") for record in stage_a_records}
    pools: Dict[str, Dict[str, Any]] = {
        STAGE_A_RANDOM: {
            "allowed_uids": stage_a_uids,
            "diagnostics": {
                "matched_pool_count": len(stage_a_uids),
                "bucket_count": 1,
                "matched_bucket_ratio": 1.0,
                "matched_selected_reference_ratio": 1.0,
                "exclude_selected": True,
                "matched_variables": [],
                "fallback_order": [],
            },
        }
    }
    for name, variables in DECOMPOSITION_ARMS:
        uids, diagnostics = _matched_bucket_baseline_pool(
            baseline_records=stage_a_records,
            selected_records=selected_records,
            bucket_fn=_bucket_fn(variables),
            seed=seed,
            pool_multiplier=pool_multiplier,
            exclude_selected=True,
        )
        diagnostics.update(
            {
                "matching_policy": name,
                "matched_variables": list(variables),
                "fallback_order": [],
            }
        )
        pools[name] = {"allowed_uids": uids, "diagnostics": diagnostics}
    return pools


def _run_dataset(dataset: str, profile_name: str, profiles_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    profile = _profile_payload(profiles_path, profile_name)
    stage_a = _stage_a_gate(profile)
    probe_cfg = _utility_probe_config(profile, evaluation_mode="certification")
    run_summary = load_json(RUN_SUMMARY_PATH)
    dataset_summary = (((run_summary.get("profiles") or {}).get(profile_name) or {}).get(dataset) or {})
    selected_path = Path(str(dataset_summary.get("output_path") or (OUTPUT_DIR / "subsets" / profile_name / f"{dataset}.jsonl")))
    scored_path = Path(str((((load_json(SCORING_MANIFEST_PATH).get("datasets") or {}).get(dataset) or {}).get("path") or "")))
    if not selected_path.exists() or not scored_path.exists():
        raise FileNotFoundError(f"Missing selected/scored input for {profile_name}:{dataset}")

    selected_records = list(iter_jsonl_records_resilient(selected_path))
    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    stage_a_records = [
        record
        for record in iter_jsonl_records_resilient(scored_path)
        if str(record.get("chunk_uid") or "") not in selected_uids and _passes_gates(record, stage_a)
    ]
    pools = _build_pools(
        stage_a_records,
        selected_records,
        seed=int(probe_cfg.get("sampling_hash_seed") or 42),
        pool_multiplier=int(((profile.get("selector") or {}).get("matched_baseline_pool_multiplier") or 4)),
    )
    _progress(f"{dataset}: selected={len(selected_records)} controls={len(stage_a_records)}")

    text_map = {str(record.get("chunk_uid") or ""): str(record.get("text") or "") for record in selected_records}
    results: Dict[str, Any] = {}
    context_cache: Dict[Any, Any] = {}
    selected_sequence_cache: Dict[Any, Any] = {}
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    try:
        for name, pool in pools.items():
            allowed_uids = set(pool["allowed_uids"])
            if not allowed_uids:
                _progress(f"{dataset}: skipping empty arm {name}")
                continue
            _progress(f"{dataset}: start {name} pool={len(allowed_uids)}")
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
                progress_label=f"matching-decomposition:{dataset}:{name}",
            )
            _progress(f"{dataset}: {name} delta_nll={results[name].get('delta_nll')}")
    finally:
        conn.close()

    return {
        "schema_version": "utility-matching-decomposition-v1",
        "profile": profile_name,
        "dataset": dataset,
        "purpose": "Identify which exact matching variable changes the Stage-C Utility estimate.",
        "selector_objective_scope": "Stage C validation only; never selector objective",
        "arm_order": list(pools),
        "probe_protocol": {
            key: probe_cfg.get(key)
            for key in ("model_name", "train_token_budget", "eval_token_budget", "bootstrap_samples", "max_train_steps", "train_epochs", "holdout_buckets", "seeds")
        },
        "selected_records": len(selected_records),
        "stage_a_control_records": len(stage_a_records),
        "pool_fidelity": {name: _pool_fidelity(name, pool["diagnostics"]) for name, pool in pools.items()},
        "result_summaries": {name: _result_summary(result) for name, result in results.items()},
        "results": results,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Utility Matching Decomposition",
        "",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Profile: `{report.get('profile')}`",
        "",
        "| Arm | Matched variables | Delta NLL | CI low | Positive runs | Matched selected | Pool |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in report.get("arm_order") or []:
        result = (report.get("result_summaries") or {}).get(name) or {}
        fidelity = (report.get("pool_fidelity") or {}).get(name) or {}
        lines.append(
            f"| {name} | {', '.join(fidelity.get('matched_variables') or []) or 'none'} | "
            f"{result.get('delta_nll')} | {result.get('delta_nll_ci_low')} | "
            f"{result.get('positive_run_fraction')} | {fidelity.get('matched_selected_reference_ratio')} | "
            f"{fidelity.get('matched_pool_count')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompose Utility matching variables under certification settings.")
    parser.add_argument("--datasets", nargs="+", default=["openwebtext2_subset"])
    parser.add_argument("--profile", default="style_taxonomy_alignment_probe")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        report = _run_dataset(str(dataset), str(args.profile), args.profiles)
        base = args.output_dir / f"utility_matching_decomposition_{dataset}"
        save_json(base.with_suffix(".json"), report)
        _write_markdown(report, base.with_suffix(".md"))
        _progress(f"wrote {base.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
