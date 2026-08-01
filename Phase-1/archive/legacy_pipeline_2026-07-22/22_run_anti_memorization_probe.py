#!/usr/bin/env python3
"""Run a focused anti-memorization Utility diagnostic baseline for one dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict

from data_eval_common import DEFAULT_PROFILE_CONFIG, OUTPUT_DIR, RUN_SUMMARY_PATH, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import (
    ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
    INDEX_DB_PATH,
    SCORING_MANIFEST_PATH,
    _anti_memorization_bucket_from_scored_record,
    _fingerprint_uids,
    _matched_bucket_baseline_pool,
    _passes_gates,
    _score_with_probe_buckets,
    _stage_a_gate,
    _stage_c_validation,
    _utility_probe_config,
)


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "validation" / "anti_memorization_probe_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "anti_memorization_probe_report.md"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "validation"


def _progress(message: str) -> None:
    print(f"[22] {message}", flush=True)


def _compact_mapping(mapping: Dict[str, Any], limit: int = 200) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in sorted(mapping.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    }


def _compact_pool_diagnostics(pool: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(pool)
    for key in ("bucket_targets", "bucket_available", "bucket_selected"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = _compact_mapping(value)
            out[f"{key}_truncated"] = len(value) > len(out[key])
            out[f"{key}_total_buckets"] = len(value)
    return out


def _profile_payload(profiles_path: Path, profile_name: str) -> Dict[str, Any]:
    payload = load_json(profiles_path)
    profiles = payload.get("profiles") or {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise RuntimeError(f"Profile {profile_name!r} not found in {profiles_path}")
    return profile


def _apply_probe_overrides(probe_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(probe_cfg)
    scalar_keys = {
        "train_token_budget": int,
        "eval_token_budget": int,
        "ood_eval_token_budget": int,
        "bootstrap_samples": int,
        "max_train_steps": int,
        "train_epochs": float,
        "train_audit_token_budget": int,
        "sampling_hash_seed": int,
        "seed": int,
        "min_probe_bucket_count": int,
    }
    for key, caster in scalar_keys.items():
        value = overrides.get(key)
        if value is not None:
            cfg[key] = caster(value)
    if overrides.get("seeds") is not None:
        cfg["seeds"] = [int(seed) for seed in overrides["seeds"]]
    if overrides.get("holdout_buckets") is not None:
        cfg["holdout_buckets"] = [int(bucket) for bucket in overrides["holdout_buckets"]]
    return cfg


def _stable_sample_records(records: list[Dict[str, Any]], *, limit: int | None, seed: int) -> list[Dict[str, Any]]:
    if limit is None or int(limit) <= 0 or len(records) <= int(limit):
        return list(records)
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha1(f"{int(seed)}:{record.get('chunk_uid') or ''}".encode("utf-8", errors="replace")).hexdigest(),
            str(record.get("chunk_uid") or ""),
        ),
    )[: int(limit)]


def _run_probe(*, dataset: str, profile_name: str, profiles_path: Path, probe_overrides: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    _progress(f"start dataset={dataset} profile={profile_name}")
    profile = _profile_payload(profiles_path, profile_name)
    stage_a = _stage_a_gate(profile)
    stage_c = _stage_c_validation(profile)
    probe_cfg = _utility_probe_config(profile, evaluation_mode=str(stage_c.get("evaluation_mode") or "development"))
    probe_cfg = _apply_probe_overrides(probe_cfg, probe_overrides)
    run_summary = load_json(RUN_SUMMARY_PATH)
    profile_summary = (run_summary.get("profiles") or {}).get(profile_name) or {}
    dataset_summary = profile_summary.get(dataset) or {}
    selected_path = Path(str(dataset_summary.get("output_path") or ""))
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected subset missing for {profile_name}:{dataset}: {selected_path}")
    scoring_manifest = load_json(SCORING_MANIFEST_PATH)
    scored_path = Path(str(((scoring_manifest.get("datasets") or {}).get(dataset) or {}).get("path") or ""))
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored file missing for {dataset}: {scored_path}")

    _progress(f"load selected start: {selected_path}")
    all_selected_records = list(iter_jsonl_records_resilient(selected_path))
    _progress(f"load selected done: records={len(all_selected_records)} elapsed={time.perf_counter() - started:.1f}s")
    all_selected_uids = {str(record.get("chunk_uid") or "") for record in all_selected_records}
    selected_records = _stable_sample_records(
        all_selected_records,
        limit=probe_overrides.get("max_selected_records"),
        seed=int(probe_cfg.get("sampling_hash_seed") or 42),
    )
    selected_uids = {str(record.get("chunk_uid") or "") for record in selected_records}
    selected_bucket_names = {
        str(_anti_memorization_bucket_from_scored_record(record) or "unknown")
        for record in selected_records
    }
    _progress(f"selected sample ready: records={len(selected_records)} buckets={len(selected_bucket_names)}")
    _progress(f"stream scored stage-a pool start: {scored_path}")
    stage_a_records = []
    scanned_records = 0
    stage_a_candidate_records = 0
    for record in iter_jsonl_records_resilient(scored_path):
        scanned_records += 1
        uid = str(record.get("chunk_uid") or "")
        if uid in all_selected_uids:
            continue
        if not _passes_gates(record, stage_a):
            continue
        stage_a_candidate_records += 1
        if str(_anti_memorization_bucket_from_scored_record(record) or "unknown") not in selected_bucket_names:
            continue
        stage_a_records.append(record)
        if scanned_records % 100000 == 0:
            _progress(
                f"stream scored progress: scanned={scanned_records} "
                f"stage_a_candidates={stage_a_candidate_records} matched_bucket_pool={len(stage_a_records)} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
    selected_scored_records = selected_records
    _progress(
        f"stage-a and selected scored ready: stage_a_pool={len(stage_a_records)} "
        f"selected_scored={len(selected_scored_records)} scanned={scanned_records} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    _progress("matched anti-mem pool build start")
    allowed_uids, pool_diagnostics = _matched_bucket_baseline_pool(
        baseline_records=stage_a_records,
        selected_records=selected_scored_records,
        bucket_fn=_anti_memorization_bucket_from_scored_record,
        seed=int(probe_cfg.get("sampling_hash_seed") or 42),
        pool_multiplier=int(((profile.get("selector") or {}).get("matched_baseline_pool_multiplier") or 4)),
        exclude_selected=False,
    )
    _progress(
        f"matched anti-mem pool build done: matched_pool={len(allowed_uids)} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    text_map = {str(record.get("chunk_uid") or ""): str(record.get("text") or "") for record in selected_records}
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    try:
        _progress(
            "utility score start: "
            f"train_tokens={probe_cfg.get('train_token_budget')} eval_tokens={probe_cfg.get('eval_token_budget')} "
            f"steps={probe_cfg.get('max_train_steps')} buckets={probe_cfg.get('holdout_buckets')} seeds={probe_cfg.get('seeds')}"
        )
        aggregate = _score_with_probe_buckets(
            conn,
            context_cache={},
            selected_sequence_cache={},
            selected_records=selected_records,
            text_map=text_map,
            baseline_variant=ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
            baseline_allowed_uids=allowed_uids,
            baseline_uid_fingerprint=_fingerprint_uids(allowed_uids),
            train_dataset=dataset,
            eval_dataset=dataset,
            probe_cfg=probe_cfg,
            eval_token_budget=int(probe_cfg["eval_token_budget"]),
            holdout_buckets=list(probe_cfg.get("holdout_buckets") or [int(probe_cfg.get("holdout_bucket") or 0)]),
            progress_label=f"{profile_name}:{dataset}:anti_memorization",
        )
        _progress(f"utility score done elapsed={time.perf_counter() - started:.1f}s")
    finally:
        conn.close()
    return {
        "schema_version": "anti-memorization-probe-report-v1",
        "profile": profile_name,
        "dataset": dataset,
        "baseline": ANTI_MEMORIZATION_DIAGNOSTIC_BASELINE,
        "purpose": "Focused Stage-C diagnostic for whether a repetition/length-matched baseline changes the Utility transfer-gap interpretation.",
        "selector_objective_scope": "diagnostic_only_not_selector_objective",
        "probe_override_scope": "diagnostic_only_not_certification",
        "probe_protocol": {
            "train_token_budget": int(probe_cfg.get("train_token_budget") or 0),
            "eval_token_budget": int(probe_cfg.get("eval_token_budget") or 0),
            "bootstrap_samples": int(probe_cfg.get("bootstrap_samples") or 0),
            "max_train_steps": int(probe_cfg.get("max_train_steps") or 0),
            "train_epochs": float(probe_cfg.get("train_epochs") or 0.0),
            "train_audit_token_budget": int(probe_cfg.get("train_audit_token_budget") or 0),
            "min_probe_bucket_count": int(probe_cfg.get("min_probe_bucket_count") or 0),
            "holdout_buckets": list(probe_cfg.get("holdout_buckets") or []),
            "seeds": list(probe_cfg.get("seeds") or []),
        },
        "selected_sample": {
            "source_selected_records": int(len(all_selected_records)),
            "diagnostic_selected_records": int(len(selected_records)),
            "max_selected_records": probe_overrides.get("max_selected_records"),
            "sampling_seed": int(probe_cfg.get("sampling_hash_seed") or 42),
            "baseline_pool_excludes_all_selected_records": True,
            "stage_a_pool_prefiltered_by_selected_anti_mem_buckets": True,
        },
        "pool_diagnostics": _compact_pool_diagnostics(pool_diagnostics),
        "utility_result": aggregate,
    }


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    utility = report.get("utility_result") or {}
    causal = utility.get("causal_utility_audit") or {}
    pool = report.get("pool_diagnostics") or {}
    lines = [
        "# Anti-Memorization Utility Probe",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Baseline: `{report.get('baseline')}`",
        f"- Matched pool records: `{pool.get('matched_pool_count')}`",
        f"- Pool bucket count: `{pool.get('bucket_count')}`",
        f"- Utility delta NLL: `{utility.get('delta_nll')}`",
        f"- Utility gain score: `{utility.get('small_lm_probe_gain_score')}`",
        f"- Delta NLL CI low: `{utility.get('delta_nll_ci_low')}`",
        f"- MDE 95 max: `{utility.get('minimum_detectable_delta_nll_95_max')}`",
        f"- Causal mode: `{causal.get('dominant_failure_mode')}`",
        f"- Train audit gap: `{causal.get('mean_selected_minus_baseline_train_audit_delta_nll')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_output_paths(
    *,
    dataset: str,
    dataset_count: int,
    output: Path | None,
    md_output: Path | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    if dataset_count == 1 and output is not None:
        json_path = output
    elif dataset_count == 1 and output is None and dataset == "wikitext103_subset":
        json_path = DEFAULT_JSON_OUTPUT
    else:
        json_path = output_dir / f"anti_memorization_probe_report_{dataset}.json"

    if dataset_count == 1 and md_output is not None:
        md_path = md_output
    elif dataset_count == 1 and md_output is None and dataset == "wikitext103_subset":
        md_path = DEFAULT_MD_OUTPUT
    else:
        md_path = output_dir / f"anti_memorization_probe_report_{dataset}.md"
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused anti-memorization Utility probe.")
    parser.add_argument("--dataset", default="wikitext103_subset")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--profile", default="canonical")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-token-budget", type=int, default=None)
    parser.add_argument("--eval-token-budget", type=int, default=None)
    parser.add_argument("--bootstrap-rounds", "--bootstrap-samples", dest="bootstrap_samples", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--train-epochs", type=float, default=None)
    parser.add_argument("--train-audit-token-budget", type=int, default=None)
    parser.add_argument("--min-probe-bucket-count", type=int, default=None)
    parser.add_argument("--holdout-buckets", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--sampling-hash-seed", type=int, default=None)
    parser.add_argument("--max-selected-records", type=int, default=None)
    args = parser.parse_args()
    datasets = [str(item) for item in (args.datasets if args.datasets is not None else [args.dataset])]
    if not datasets:
        raise RuntimeError("No datasets requested.")
    if len(datasets) > 1 and (args.output is not None or args.md_output is not None):
        raise RuntimeError("--output/--md-output are only supported for a single dataset; use --output-dir for multiple datasets.")
    probe_overrides = {
        "train_token_budget": args.train_token_budget,
        "eval_token_budget": args.eval_token_budget,
        "ood_eval_token_budget": args.eval_token_budget,
        "bootstrap_samples": args.bootstrap_samples,
        "max_train_steps": args.max_train_steps,
        "train_epochs": args.train_epochs,
        "train_audit_token_budget": args.train_audit_token_budget,
        "min_probe_bucket_count": args.min_probe_bucket_count,
        "holdout_buckets": args.holdout_buckets,
        "seeds": args.seeds,
        "sampling_hash_seed": args.sampling_hash_seed,
        "max_selected_records": args.max_selected_records,
    }
    for dataset in datasets:
        json_path, md_path = _dataset_output_paths(
            dataset=dataset,
            dataset_count=len(datasets),
            output=args.output,
            md_output=args.md_output,
            output_dir=args.output_dir,
        )
        report = _run_probe(
            dataset=dataset,
            profile_name=str(args.profile),
            profiles_path=args.profiles,
            probe_overrides=probe_overrides,
        )
        save_json(json_path, report)
        _write_markdown(report, md_path)
        print(f"[22] anti-memorization probe json: {json_path}", flush=True)
        print(f"[22] anti-memorization probe md: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
