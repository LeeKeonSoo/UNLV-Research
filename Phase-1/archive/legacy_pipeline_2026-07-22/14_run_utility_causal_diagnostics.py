#!/usr/bin/env python3
"""Run the official Utility sensitivity audit without changing pipeline outputs.

This script answers four questions before further selector tuning:
1. Can the small-LM Utility probe separate positive and negative controls?
2. Is the current failure already present in-domain, or mainly caused by OOD enforcement?
3. How strong is the canonical multi-matched baseline relative to weaker baselines?
4. Are selected subsets weak because of learnability, or because the probe/protocol is insensitive?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from data_eval_common import (
    INDEX_DIR,
    OUTPUT_DIR,
    RUN_MANIFEST_PATH,
    RUN_SUMMARY_PATH,
    SCORED_DIR,
    iter_jsonl_records_resilient,
    save_json,
)
from policy.subsets import _objective_components
from utility.lm_probe import build_probe_context, score_selected_records

INDEX_DB_PATH = INDEX_DIR / "index.sqlite"
DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "validation" / "utility_sensitivity_audit.json"
LEGACY_OUTPUT_PATH = OUTPUT_DIR / "validation" / "utility_causal_diagnostics.json"
CANONICAL_BASELINE = "baseline_stageA_random"


def _stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def _metric_score(record: Dict[str, Any], metric: str) -> float:
    payload = (record.get("core_metrics") or {}).get(metric) or {}
    try:
        return float(payload.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stage_a_pass(record: Dict[str, Any]) -> bool:
    return bool(
        _metric_score(record, "structural_validity_gate") >= 1.0
        and _metric_score(record, "exact_duplicate_indicator") <= 0.0
        and _metric_score(record, "shingle_near_duplicate_indicator") <= 0.0
    )


def _quality(record: Dict[str, Any]) -> float:
    return _metric_score(record, "reference_quality_score")


def _risk(record: Dict[str, Any]) -> float:
    return _metric_score(record, "shingle_near_duplicate_risk_score")


def _learnability(record: Dict[str, Any]) -> float:
    try:
        return float(_objective_components(record).get("learnability_support") or 0.0)
    except Exception:
        return 0.0


def _load_scored(dataset: str) -> List[Dict[str, Any]]:
    path = SCORED_DIR / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return list(iter_jsonl_records_resilient(path))


def _load_selected_uids(run_summary: Dict[str, Any], profile: str, dataset: str) -> set[str]:
    profile_payload = (run_summary.get("profiles") or {}).get(profile) or {}
    meta = profile_payload.get(dataset) or {}
    path = Path(str(meta.get("output_path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"selected subset missing for {profile}:{dataset}: {path}")
    return {str(record.get("chunk_uid") or "") for record in iter_jsonl_records_resilient(path)}


def _arm_records(
    *,
    dataset: str,
    scored_records: List[Dict[str, Any]],
    selected_uids: set[str],
    arm: str,
    arm_pool_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    stage_a = [record for record in scored_records if _stage_a_pass(record)]
    if not stage_a:
        raise RuntimeError(f"{dataset}: no Stage-A records available for Utility diagnostics.")
    limit = min(max(1, int(arm_pool_size)), len(stage_a))
    if arm == "selected":
        records = [record for record in scored_records if str(record.get("chunk_uid") or "") in selected_uids and _stage_a_pass(record)]
        records.sort(key=lambda record: (_stable_unit(str(record.get("chunk_uid") or ""), seed), str(record.get("chunk_uid") or "")))
        return records[:limit]
    if arm == "positive_control":
        records = sorted(
            stage_a,
            key=lambda record: (
                -_learnability(record),
                -_quality(record),
                _risk(record),
                str(record.get("chunk_uid") or ""),
            ),
        )
        return records[:limit]
    if arm in {
        "negative_control",
        "low_quality_negative_control",
        "corrupted_negative_control",
        "token_shuffle_negative_control",
        "destructive_negative_control",
    }:
        records = sorted(
            stage_a,
            key=lambda record: (
                _quality(record),
                -_risk(record),
                _learnability(record),
                str(record.get("chunk_uid") or ""),
            ),
        )
        return records[:limit]
    if arm == "stageA_random":
        records = sorted(stage_a, key=lambda record: (_stable_unit(str(record.get("chunk_uid") or ""), seed), str(record.get("chunk_uid") or "")))
        return records[:limit]
    raise ValueError(f"Unknown diagnostic arm: {arm}")


def _effective_arm_pool_size(stage_a_count: int, arm_count: int, requested: int) -> int:
    requested = max(1, int(requested))
    stage_a_count = max(0, int(stage_a_count))
    arm_count = max(1, int(arm_count))
    if stage_a_count <= 1:
        return 1
    comparable_baseline_cap = max(1, stage_a_count // (arm_count + 1))
    non_empty_baseline_cap = max(1, (stage_a_count - 1) // arm_count)
    return max(1, min(requested, comparable_baseline_cap, non_empty_baseline_cap))


def _fetch_text_pairs(conn: sqlite3.Connection, dataset: str, records: Sequence[Dict[str, Any]]) -> List[tuple[str, str]]:
    uids = [str(record.get("chunk_uid") or "") for record in records if str(record.get("chunk_uid") or "")]
    if not uids:
        return []
    text_by_uid: Dict[str, str] = {}
    for i in range(0, len(uids), 800):
        batch = uids[i : i + 800]
        placeholders = ",".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT chunk_uid, text FROM chunks WHERE dataset = ? AND chunk_uid IN ({placeholders})",
            [str(dataset), *batch],
        ).fetchall()
        text_by_uid.update({str(uid): str(text or "") for uid, text in rows})
    return [(uid, text_by_uid.get(uid, "")) for uid in uids if uid in text_by_uid]


def _corrupt_text_for_negative_control(text: str, *, uid: str, seed: int, mode: str) -> str:
    """Destroy next-token structure while preserving rough token inventory.

    A same-dataset low-quality sample can still improve NLL because it shares
    vocabulary and textbook phrasing. This control is deliberately corrupted so
    probe validity tests whether the LM distinguishes learnable sequence
    structure, not just domain-token exposure.
    """
    tokens = str(text or "").split()
    if len(tokens) < 4:
        return " ".join(reversed(list(str(text or "")))) or "<corrupted>"
    digest = hashlib.sha1(f"{int(seed)}:{uid}:corrupt".encode("utf-8", errors="replace")).hexdigest()
    rng = np.random.default_rng(int(digest[:16], 16))
    if mode == "hash_noise":
        alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz"), dtype="<U1")
        noisy_tokens = []
        for token in tokens:
            length = min(18, max(3, len(str(token))))
            chars = rng.choice(alphabet, size=length, replace=True)
            noisy_tokens.append("".join(chars.tolist()))
        return " ".join(noisy_tokens)
    if mode == "char_shuffle":
        corrupted = []
        for token in tokens:
            chars = list(str(token))
            if len(chars) > 3:
                rng.shuffle(chars)
            corrupted.append("".join(chars))
        rng.shuffle(corrupted)
        return " ".join(corrupted)
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def _maybe_transform_arm_pairs(
    arm: str,
    pairs: List[tuple[str, str]],
    *,
    seed: int,
    corruption_mode: str,
) -> List[tuple[str, str]]:
    if arm not in {"corrupted_negative_control", "destructive_negative_control", "token_shuffle_negative_control"}:
        return pairs
    mode = "token_shuffle" if arm == "token_shuffle_negative_control" else corruption_mode
    if arm == "destructive_negative_control":
        mode = "hash_noise"
    return [
        (f"{uid}::corrupted", _corrupt_text_for_negative_control(text, uid=uid, seed=seed, mode=mode))
        for uid, text in pairs
    ]


def _record_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"records": 0}
    qualities = [_quality(record) for record in records]
    risks = [_risk(record) for record in records]
    learns = [_learnability(record) for record in records]
    words = [int(record.get("word_count") or 0) for record in records]
    return {
        "records": int(len(records)),
        "mean_quality": round(float(np.mean(qualities)), 6),
        "mean_redundancy_risk": round(float(np.mean(risks)), 6),
        "mean_learnability_support": round(float(np.mean(learns)), 6),
        "mean_word_count": round(float(np.mean(words)), 3),
    }


def _score_arm(
    *,
    conn: sqlite3.Connection,
    dataset: str,
    arm: str,
    arm_records: List[Dict[str, Any]],
    stage_a_uids: set[str],
    common_baseline_allowed_uids: set[str] | None,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    arm_uids = {str(record.get("chunk_uid") or "") for record in arm_records}
    baseline_allowed_uids = set(common_baseline_allowed_uids) if common_baseline_allowed_uids is not None else set(stage_a_uids) - arm_uids
    baseline_fingerprint = hashlib.sha1("\n".join(sorted(baseline_allowed_uids)).encode()).hexdigest()
    pairs = _fetch_text_pairs(conn, dataset, arm_records)
    if not pairs:
        raise RuntimeError(f"{dataset}:{arm}: no text pairs available")
    pairs = _maybe_transform_arm_pairs(
        str(arm),
        pairs,
        seed=int(args.seed),
        corruption_mode=str(args.corruption_mode),
    )
    context = build_probe_context(
        conn,
        baseline_variant=CANONICAL_BASELINE,
        baseline_allowed_uids=baseline_allowed_uids,
        baseline_uid_fingerprint=baseline_fingerprint,
        dataset=dataset,
        eval_dataset=dataset,
        token_budget=int(args.train_token_budget),
        eval_token_budget=int(args.eval_token_budget),
        holdout_modulo=int(args.holdout_modulo),
        holdout_bucket=int(args.holdout_bucket),
        sampling_hash_seed=int(args.sampling_hash_seed),
        model_name=str(args.model_name),
        max_length=int(args.max_length),
        train_batch_size=int(args.train_batch_size),
        eval_batch_size=int(args.eval_batch_size),
        learning_rate=float(args.learning_rate),
        max_train_steps=int(args.max_train_steps),
        train_epochs=float(args.train_epochs),
        train_audit_token_budget=int(args.train_audit_token_budget),
    )
    result = score_selected_records(
        context,
        pairs,
        bootstrap_rounds=int(args.bootstrap_rounds),
        seed=int(args.seed),
        selected_fingerprint=f"diagnostic:{dataset}:{arm}:{len(pairs)}:{args.seed}",
        selected_sequence_cache={},
    )
    causal = result.get("causal_failure_mode")
    return {
        "arm": arm,
        "arm_summary": _record_summary(arm_records),
        "baseline_pool_records": int(len(baseline_allowed_uids)),
        "baseline_uid_fingerprint": baseline_fingerprint,
        "baseline_policy": (
            "common_stageA_baseline_disjoint_from_all_sensitivity_arms"
            if common_baseline_allowed_uids is not None
            else "legacy_stageA_baseline_disjoint_from_current_arm_only"
        ),
        "corruption_mode": (
            "token_shuffle"
            if arm == "token_shuffle_negative_control"
            else "hash_noise"
            if arm == "destructive_negative_control"
            else str(args.corruption_mode)
            if arm == "corrupted_negative_control"
            else None
        ),
        "small_lm_probe_gain_score": result.get("small_lm_probe_gain_score"),
        "delta_nll": result.get("delta_nll"),
        "delta_nll_ci_low": result.get("delta_nll_ci_low"),
        "minimum_detectable_delta_nll_95": result.get("minimum_detectable_delta_nll_95"),
        "effect_to_mde_ratio": result.get("effect_to_mde_ratio"),
        "causal_failure_mode": causal,
        "selected_train_audit_delta_nll": result.get("selected_train_audit_delta_nll"),
        "baseline_train_audit_delta_nll": result.get("baseline_train_audit_delta_nll"),
        "selected_minus_baseline_train_audit_delta_nll": result.get("selected_minus_baseline_train_audit_delta_nll"),
        "probe_device": result.get("probe_device"),
        "train_tokens": {
            "arm": result.get("selected_train_tokens"),
            "baseline": result.get("baseline_train_tokens"),
        },
        "eval_tokens": result.get("eval_tokens"),
    }


def _arm_delta(by_arm: Dict[str, Dict[str, Any]], arm: str | None) -> float | None:
    if not arm:
        return None
    value = (by_arm.get(str(arm)) or {}).get("delta_nll")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _destructive_negative_arm(by_arm: Dict[str, Dict[str, Any]]) -> str | None:
    if "destructive_negative_control" in by_arm:
        return "destructive_negative_control"
    corrupted = by_arm.get("corrupted_negative_control")
    if corrupted and str(corrupted.get("corruption_mode") or "") != "token_shuffle":
        return "corrupted_negative_control"
    if "negative_control" in by_arm:
        return "negative_control"
    if corrupted:
        return "corrupted_negative_control"
    return None


def _token_inventory_stress_arm(by_arm: Dict[str, Dict[str, Any]]) -> str | None:
    if "token_shuffle_negative_control" in by_arm:
        return "token_shuffle_negative_control"
    corrupted = by_arm.get("corrupted_negative_control")
    if corrupted and str(corrupted.get("corruption_mode") or "") == "token_shuffle":
        return "corrupted_negative_control"
    return None


def _arm_mde(by_arm: Dict[str, Dict[str, Any]], arm: str | None) -> float | None:
    if not arm:
        return None
    value = (by_arm.get(str(arm)) or {}).get("minimum_detectable_delta_nll_95")
    if value is None:
        return None
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return None


def _pairwise_margin_status(
    *,
    left: float | None,
    right: float | None,
    left_mde: float | None,
    right_mde: float | None,
) -> Dict[str, Any]:
    if left is None or right is None:
        return {
            "margin": None,
            "mde": None,
            "left_gt_right": False,
            "decisive": False,
            "near_noise_floor": False,
        }
    margin = float(left) - float(right)
    mde_values = [float(value) for value in (left_mde, right_mde) if value is not None]
    pairwise_mde = max(mde_values) if mde_values else None
    decisive = bool(pairwise_mde is not None and abs(margin) > pairwise_mde)
    return {
        "margin": round(margin, 8),
        "mde": None if pairwise_mde is None else round(float(pairwise_mde), 8),
        "left_gt_right": bool(margin > 0.0),
        "decisive": decisive,
        "near_noise_floor": bool(pairwise_mde is not None and abs(margin) <= pairwise_mde),
    }


def _probe_sensitivity_order(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = {str(result.get("arm")): result for result in arm_results}
    positive = _arm_delta(by_arm, "positive_control")
    random = _arm_delta(by_arm, "stageA_random")
    selected = _arm_delta(by_arm, "selected")
    negative_arm = _destructive_negative_arm(by_arm)
    negative = _arm_delta(by_arm, negative_arm)
    token_stress_arm = _token_inventory_stress_arm(by_arm)
    token_stress = _arm_delta(by_arm, token_stress_arm)
    positive_relation = _pairwise_margin_status(
        left=positive,
        right=random,
        left_mde=_arm_mde(by_arm, "positive_control"),
        right_mde=_arm_mde(by_arm, "stageA_random"),
    )
    negative_relation = _pairwise_margin_status(
        left=random,
        right=negative,
        left_mde=_arm_mde(by_arm, "stageA_random"),
        right_mde=_arm_mde(by_arm, negative_arm),
    )
    selected_relation = _pairwise_margin_status(
        left=selected,
        right=random,
        left_mde=_arm_mde(by_arm, "selected"),
        right_mde=_arm_mde(by_arm, "stageA_random"),
    )
    token_relation = _pairwise_margin_status(
        left=random,
        right=token_stress,
        left_mde=_arm_mde(by_arm, "stageA_random"),
        right_mde=_arm_mde(by_arm, token_stress_arm),
    )
    positive_gt_random = bool(positive_relation["left_gt_right"])
    random_gt_negative = bool(negative_relation["left_gt_right"])
    destructive_probe_valid = bool(positive_gt_random and random_gt_negative)
    selected_gt_random = bool(selected_relation["left_gt_right"])
    token_inventory_stress_pass = None if token_stress is None or random is None else bool(token_relation["left_gt_right"])
    token_exposure_confounded = bool(
        token_inventory_stress_pass is False
        and token_relation.get("decisive")
    )
    token_exposure_inconclusive = bool(
        token_inventory_stress_pass is False
        and token_relation.get("near_noise_floor")
    )
    if negative is None:
        utility_evidence_status = "probe_not_evaluable"
    elif not positive_gt_random:
        utility_evidence_status = (
            "positive_control_inconclusive_near_noise_floor"
            if positive_relation.get("near_noise_floor")
            else "positive_control_not_separated"
        )
    elif not random_gt_negative:
        utility_evidence_status = (
            "destructive_negative_inconclusive_near_noise_floor"
            if negative_relation.get("near_noise_floor")
            else "destructive_negative_not_separated"
        )
    elif token_exposure_confounded:
        utility_evidence_status = "probe_valid_token_exposure_confounded"
    elif token_exposure_inconclusive:
        utility_evidence_status = "probe_valid_token_exposure_inconclusive"
    else:
        utility_evidence_status = "probe_valid"
    return {
        "expected_order": f"positive_control > stageA_random > {negative_arm}",
        "canonical_negative_control": negative_arm,
        "destructive_negative_control": negative_arm,
        "token_inventory_stress_control": token_stress_arm,
        "probe_valid": bool(destructive_probe_valid),
        "destructive_probe_valid": bool(destructive_probe_valid),
        "positive_gt_random": bool(positive_gt_random),
        "random_gt_negative": bool(random_gt_negative),
        "random_gt_destructive_negative": bool(random_gt_negative),
        "selected_gt_random": bool(selected_gt_random),
        "order_pass": bool(destructive_probe_valid),
        "destructive_order_pass": bool(destructive_probe_valid),
        "token_inventory_stress_pass": token_inventory_stress_pass,
        "token_exposure_confounded": bool(token_exposure_confounded),
        "token_exposure_inconclusive": bool(token_exposure_inconclusive),
        "control_margins": {
            "positive_minus_random": positive_relation,
            "random_minus_destructive_negative": negative_relation,
            "selected_minus_random": selected_relation,
            "random_minus_token_inventory_stress": token_relation,
        },
        "utility_evidence_status": utility_evidence_status,
        "delta_nll_by_arm": {
            arm: result.get("delta_nll") for arm, result in sorted(by_arm.items())
        },
        "interpretation": (
            "Destructive probe validity tests positive_control > stageA_random > destructive_negative_control. "
            "Token-shuffle stress is reported separately as a token-exposure confounding caveat. "
            "If destructive_probe_valid is true but selected_gt_random is false, Stage C is sensitive enough to flag a validation gap; "
            "Utility remains diagnostic-only and must not become a selector objective."
        ),
    }


def _root_cause_decision(probe_sensitivity: Dict[str, Any], current_failure: Dict[str, Any]) -> Dict[str, Any]:
    destructive_probe_valid = bool(probe_sensitivity.get("destructive_probe_valid", probe_sensitivity.get("order_pass")))
    selected_gt_random = bool(probe_sensitivity.get("selected_gt_random"))
    token_exposure_confounded = bool(probe_sensitivity.get("token_exposure_confounded"))
    token_exposure_inconclusive = bool(probe_sensitivity.get("token_exposure_inconclusive"))
    token_exposure_caveat = bool(token_exposure_confounded or token_exposure_inconclusive)
    in_domain_fails = bool(current_failure.get("in_domain_already_fails"))
    if not destructive_probe_valid:
        primary = "probe_or_protocol_sensitivity_unverified"
        action = "Do not tune the selector further until the Utility probe separates positive/random/destructive-negative controls with enough budget."
    elif in_domain_fails and not selected_gt_random:
        primary = "stage_c_selected_vs_random_gap"
        action = "The probe is sensitive and the selected subset does not beat random in this diagnostic; hold selector/Core claims and run Stage-C follow-up without adding Utility to the selector objective."
    elif not in_domain_fails:
        primary = "ood_or_transfer_bottleneck"
        action = "In-domain Utility is not the main failure; analyze OOD transfer pairs and certification scope."
    elif token_exposure_caveat:
        primary = "selector_candidate_with_token_exposure_caveat"
        action = "Selected beats random under a destructive sanity check, but token-shuffle stress adds a token-exposure caveat; report Utility evidence with this caveat."
    else:
        primary = "baseline_strength_or_small_effect_bottleneck"
        action = "Probe is sensitive but selected is close to random; inspect canonical baseline strength and effect size."
    return {
        "primary_hypothesis": primary,
        "recommended_action": action,
        "selector_tuning_allowed": False,
        "selector_policy_action": "hold",
        "utility_scope": "Stage C diagnostic only; never selector objective",
        "selector_tuning_caveat": (
            "token_exposure_confounded"
            if token_exposure_confounded
            else "token_exposure_inconclusive"
            if token_exposure_inconclusive
            else None
        ),
        "utility_failure_class": (
            "probe_not_evaluable"
            if not destructive_probe_valid
            else "selector_candidate_with_token_exposure_caveat"
            if token_exposure_caveat
            else "selector_candidate"
            if in_domain_fails and not selected_gt_random
            else "strict_counterfactual_or_transfer_candidate"
        ),
        "reasoning": {
            "probe_order_pass": destructive_probe_valid,
            "destructive_probe_valid": destructive_probe_valid,
            "utility_probe_valid": destructive_probe_valid,
            "selected_gt_random": selected_gt_random,
            "token_exposure_confounded": token_exposure_confounded,
            "token_exposure_inconclusive": token_exposure_inconclusive,
            "current_in_domain_already_fails": in_domain_fails,
        },
    }


def _current_run_failure_decomposition(run_summary: Dict[str, Any], profile: str, dataset: str) -> Dict[str, Any]:
    meta = ((run_summary.get("profiles") or {}).get(profile) or {}).get(dataset) or {}
    utility = meta.get("utility_probe_details") or {}
    aggregate = utility.get("aggregate") or {}
    evidence = aggregate.get("utility_evidence_summary") or {}
    in_domain = (utility.get("in_domain") or {}).get("baseline_multi_matched_stageA_random") or {}
    ood = utility.get("out_of_domain") or {}
    ood_cells = {}
    for eval_dataset, baselines in ood.items():
        cell = (baselines or {}).get("baseline_multi_matched_stageA_random") or {}
        ood_cells[str(eval_dataset)] = {
            "small_lm_probe_gain_score": cell.get("small_lm_probe_gain_score"),
            "delta_nll": cell.get("delta_nll"),
            "delta_nll_min": cell.get("delta_nll_min"),
            "delta_nll_ci_low": cell.get("delta_nll_ci_low"),
            "causal_mode": (cell.get("causal_utility_audit") or {}).get("dominant_failure_mode"),
        }
    return {
        "current_selected_utility": meta.get("small_lm_probe_gain_score"),
        "current_stage_c_pass": (meta.get("stage_c_core_validation") or {}).get("passed"),
        "current_signal_status": evidence.get("signal_status"),
        "in_domain_already_fails": bool(float(in_domain.get("delta_nll_min") or in_domain.get("delta_nll") or 0.0) <= 0.0),
        "in_domain": {
            "small_lm_probe_gain_score": in_domain.get("small_lm_probe_gain_score"),
            "small_lm_probe_gain_score_min": in_domain.get("small_lm_probe_gain_score_min"),
            "delta_nll": in_domain.get("delta_nll"),
            "delta_nll_min": in_domain.get("delta_nll_min"),
            "delta_nll_ci_low": in_domain.get("delta_nll_ci_low"),
            "causal_mode": (in_domain.get("causal_utility_audit") or {}).get("dominant_failure_mode"),
        },
        "ood": ood_cells,
        "baseline_minima": aggregate.get("baseline_minima"),
    }


def _resolve_profile(run_summary: Dict[str, Any], requested_profile: str) -> str:
    profiles = run_summary.get("profiles") or {}
    if requested_profile in profiles:
        return requested_profile
    profile_names = [
        str(name)
        for name, payload in profiles.items()
        if not str(name).startswith("_") and isinstance(payload, dict)
    ]
    if len(profile_names) == 1:
        fallback = profile_names[0]
        print(
            f"[14] requested profile={requested_profile!r} not found; using only available profile={fallback!r}",
            flush=True,
        )
        return fallback
    raise RuntimeError(
        f"Requested profile {requested_profile!r} not found. Available profiles: {profile_names}"
    )


def _status_from_audit(dataset: str, audit_payload: Dict[str, Any]) -> Dict[str, Any]:
    sensitivity = audit_payload.get("probe_sensitivity") or {}
    root = audit_payload.get("root_cause_decision") or {}
    if not isinstance(sensitivity, dict):
        sensitivity = {}
    if not isinstance(root, dict):
        root = {}
    probe_valid = sensitivity.get("destructive_probe_valid", sensitivity.get("probe_valid", sensitivity.get("order_pass")))
    status = str(sensitivity.get("utility_evidence_status") or ("probe_valid" if bool(probe_valid) else "probe_not_evaluable"))
    return {
        "available": True,
        "probe_valid": bool(probe_valid),
        "destructive_probe_valid": bool(probe_valid),
        "status": status,
        "selector_tuning_allowed": False,
        "selector_tuning_caveat": root.get("selector_tuning_caveat"),
        "source": str(DEFAULT_OUTPUT_PATH),
        "expected_order": sensitivity.get("expected_order"),
        "positive_gt_random": bool(sensitivity.get("positive_gt_random")),
        "random_gt_negative": bool(sensitivity.get("random_gt_negative", sensitivity.get("random_gt_destructive_negative"))),
        "random_gt_destructive_negative": bool(sensitivity.get("random_gt_destructive_negative", sensitivity.get("random_gt_negative"))),
        "selected_gt_random": bool(sensitivity.get("selected_gt_random")),
        "token_inventory_stress_pass": sensitivity.get("token_inventory_stress_pass"),
        "token_exposure_confounded": bool(sensitivity.get("token_exposure_confounded")),
        "token_exposure_inconclusive": bool(sensitivity.get("token_exposure_inconclusive")),
        "control_margins": sensitivity.get("control_margins") or {},
        "canonical_negative_control": sensitivity.get("canonical_negative_control"),
        "destructive_negative_control": sensitivity.get("destructive_negative_control"),
        "token_inventory_stress_control": sensitivity.get("token_inventory_stress_control"),
        "delta_nll_by_arm": sensitivity.get("delta_nll_by_arm") or {},
                "root_cause": root.get("primary_hypothesis"),
                "selector_policy_action": root.get("selector_policy_action", "hold"),
                "utility_scope": root.get("utility_scope", "Stage C diagnostic only; never selector objective"),
        "dataset": str(dataset),
    }


def _evidence_tier_from_statuses(
    probe_status: Dict[str, Any],
    curation_status: Dict[str, Any],
    strict_status: Dict[str, Any],
) -> str:
    if probe_status.get("probe_valid") is False:
        return "not_evaluable_utility_evidence"
    strict_name = str(strict_status.get("status") or "")
    curation_name = str(curation_status.get("status") or "")
    if strict_name == "strict_certification_ready":
        return "strict_certification_ready"
    if strict_name == "matched_baseline_gain":
        return "matched_baseline_gain"
    token_exposure_caveat = bool(probe_status.get("token_exposure_confounded") or probe_status.get("token_exposure_inconclusive"))
    if curation_name == "random_baseline_gain":
        return "random_baseline_gain_with_token_exposure_caveat" if token_exposure_caveat else "random_baseline_gain"
    if token_exposure_caveat:
        return "probe_valid_token_exposure_caveat"
    return "matched_baseline_inconclusive"


def _failure_reason_from_statuses(
    probe_status: Dict[str, Any],
    curation_status: Dict[str, Any],
    strict_status: Dict[str, Any],
) -> str:
    if probe_status.get("probe_valid") is False:
        return "probe_not_evaluable"
    strict_name = str(strict_status.get("status") or "")
    curation_name = str(curation_status.get("status") or "")
    if strict_name in {"strict_certification_ready", "matched_baseline_gain"}:
        return "pass"
    token_exposure_caveat = bool(probe_status.get("token_exposure_confounded") or probe_status.get("token_exposure_inconclusive"))
    if curation_name == "random_baseline_gain":
        return "random_gain_only_with_token_exposure_caveat" if token_exposure_caveat else "random_gain_only"
    if strict_name == "matched_baseline_inconclusive":
        return "matched_inconclusive"
    if probe_status.get("selected_gt_random") is False:
        return "selected_below_stageA_random"
    return "strict_negative"


def _profile_dataset_map(payload: Dict[str, Any], profile: str) -> Dict[str, Any]:
    profile_payload = (payload.get("profiles") or {}).get(profile) or {}
    datasets = profile_payload.get("datasets") if isinstance(profile_payload, dict) else None
    if isinstance(datasets, dict):
        return datasets
    return profile_payload if isinstance(profile_payload, dict) else {}


def _apply_sensitivity_audit_to_payload(payload: Dict[str, Any], audit: Dict[str, Any]) -> bool:
    profile = str(audit.get("profile") or "")
    datasets = audit.get("datasets") or {}
    if not profile or not isinstance(datasets, dict):
        return False
    profile_datasets = _profile_dataset_map(payload, profile)
    changed = False
    for dataset, audit_payload in datasets.items():
        if not isinstance(audit_payload, dict):
            continue
        meta = profile_datasets.get(str(dataset))
        if not isinstance(meta, dict):
            continue
        aggregate = ((meta.get("utility_probe_details") or {}).get("aggregate") or {})
        evidence = aggregate.get("utility_evidence_summary") or {}
        if not isinstance(evidence, dict):
            evidence = {}
            aggregate["utility_evidence_summary"] = evidence
        probe_status = _status_from_audit(str(dataset), audit_payload)
        curation_status = evidence.get("curation_benefit_status") or aggregate.get("curation_benefit_status") or {}
        strict_status = evidence.get("strict_counterfactual_status") or aggregate.get("strict_counterfactual_status") or {}
        if not isinstance(curation_status, dict):
            curation_status = {}
        if not isinstance(strict_status, dict):
            strict_status = {}
        evidence_tier = _evidence_tier_from_statuses(probe_status, curation_status, strict_status)
        failure_reason = _failure_reason_from_statuses(probe_status, curation_status, strict_status)
        utility_probe_valid = bool(probe_status.get("probe_valid"))
        utility_strict_pass = bool(
            utility_probe_valid
            and strict_status.get("status") in {"strict_certification_ready", "matched_baseline_gain"}
        )
        evidence.update(
            {
                "utility_probe_valid": utility_probe_valid,
                "utility_strict_pass": utility_strict_pass,
                "probe_sensitivity_status": probe_status,
                "evidence_tier": evidence_tier,
                "failure_reason": failure_reason,
            }
        )
        aggregate.update(
            {
                "utility_probe_valid": utility_probe_valid,
                "utility_strict_pass": utility_strict_pass,
                "probe_sensitivity_status": probe_status,
                "evidence_tier": evidence_tier,
                "utility_failure_reason": failure_reason,
                "utility_evidence_summary": evidence,
            }
        )
        failure_analysis = aggregate.get("utility_failure_analysis") or {}
        if isinstance(failure_analysis, dict):
            failure_analysis.update(
                {
                    "evidence_aware_failure_reason": failure_reason,
                    "probe_sensitivity_status": probe_status,
                }
            )
            aggregate["utility_failure_analysis"] = failure_analysis
        stage_c = meta.get("stage_c_core_validation") or {}
        if isinstance(stage_c, dict):
            stage_c.update(
                {
                    "utility_probe_valid": utility_probe_valid,
                    "utility_strict_pass": utility_strict_pass,
                    "utility_failure_reason": failure_reason,
                    "utility_probe_sensitivity_status": probe_status,
                    "utility_evidence_tier": evidence_tier,
                }
            )
            meta["stage_c_core_validation"] = stage_c
        core_axes = meta.get("core_axes") or {}
        if isinstance(core_axes, dict) and isinstance(core_axes.get("utility"), dict):
            core_axes["utility"]["details"] = aggregate
        changed = True
    return changed


def _update_run_artifacts_with_audit(audit: Dict[str, Any]) -> None:
    for path in (RUN_SUMMARY_PATH, RUN_MANIFEST_PATH):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if _apply_sensitivity_audit_to_payload(payload, audit):
            save_json(path, payload)
            print(f"[14] updated Utility sensitivity fields in {path}", flush=True)


def _reanalyze_existing_audit(audit: Dict[str, Any], run_summary: Dict[str, Any]) -> Dict[str, Any]:
    profile = _resolve_profile(run_summary, str(audit.get("profile") or "canonical"))
    audit["profile"] = profile
    protocol = audit.get("protocol")
    if isinstance(protocol, dict):
        protocol.setdefault("sensitivity_design", "two_stage_destructive_plus_token_inventory_stress")
        protocol.setdefault("baseline_policy", "common_stageA_baseline_disjoint_from_all_sensitivity_arms")
    datasets = audit.get("datasets") or {}
    if not isinstance(datasets, dict):
        raise RuntimeError("Existing audit has no datasets payload to reanalyze.")
    for dataset, payload in datasets.items():
        if not isinstance(payload, dict):
            continue
        arm_results = payload.get("arm_results") or []
        if not isinstance(arm_results, list) or not arm_results:
            raise RuntimeError(f"{dataset}: existing audit has no arm_results to reanalyze.")
        current_failure = payload.get("current_run_failure_decomposition")
        if not isinstance(current_failure, dict):
            current_failure = _current_run_failure_decomposition(run_summary, profile, str(dataset))
            payload["current_run_failure_decomposition"] = current_failure
        probe_sensitivity = _probe_sensitivity_order(arm_results)
        payload["probe_sensitivity"] = probe_sensitivity
        payload["root_cause_decision"] = _root_cause_decision(probe_sensitivity, current_failure)
        payload["selector_tuning_allowed"] = False
        payload["selector_policy_action"] = "hold"
        payload["utility_scope"] = "Stage C diagnostic only; never selector objective"
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the official Utility sensitivity audit.")
    parser.add_argument("--profile", default="paper_release_certification")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument(
        "--arms",
        nargs="*",
        default=[
            "selected",
            "positive_control",
            "stageA_random",
            "corrupted_negative_control",
            "token_shuffle_negative_control",
            "low_quality_negative_control",
        ],
    )
    parser.add_argument("--arm-pool-size", type=int, default=20000)
    parser.add_argument("--train-token-budget", type=int, default=12000)
    parser.add_argument("--eval-token-budget", type=int, default=6000)
    parser.add_argument("--bootstrap-rounds", type=int, default=80)
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-train-steps", type=int, default=96)
    parser.add_argument("--train-epochs", type=float, default=1.0)
    parser.add_argument("--train-audit-token-budget", type=int, default=2048)
    parser.add_argument(
        "--corruption-mode",
        choices=("token_shuffle", "char_shuffle", "hash_noise"),
        default="hash_noise",
        help="Transformation used for corrupted_negative_control; token_shuffle_negative_control is always token-shuffled.",
    )
    parser.add_argument("--holdout-modulo", type=int, default=17)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    parser.add_argument("--sampling-hash-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--update-summary-only",
        action="store_true",
        help="Do not rerun probes; apply an existing sensitivity audit to run_summary/run_manifest.",
    )
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="Do not rerun probes; recompute sensitivity taxonomy from existing arm_results, then update artifacts.",
    )
    parser.add_argument(
        "--no-update-artifacts",
        action="store_true",
        help="Write the audit file but do not mutate run_summary.json or run_manifest.json. Use for sweeps.",
    )
    args = parser.parse_args()

    run_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    if args.reanalyze_existing:
        audit = json.loads(args.output.read_text(encoding="utf-8"))
        audit = _reanalyze_existing_audit(audit, run_summary)
        args.output.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.output == DEFAULT_OUTPUT_PATH:
            LEGACY_OUTPUT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.no_update_artifacts:
            print("[14] skipped run_summary/run_manifest update (--no-update-artifacts)", flush=True)
        else:
            _update_run_artifacts_with_audit(audit)
        print(f"[14] reanalyzed existing utility sensitivity audit: {args.output}")
        return 0
    if args.update_summary_only:
        audit = json.loads(args.output.read_text(encoding="utf-8"))
        _update_run_artifacts_with_audit(audit)
        return 0

    args.profile = _resolve_profile(run_summary, str(args.profile))
    profile_payload = (run_summary.get("profiles") or {}).get(args.profile) or {}
    datasets = args.datasets or [name for name in profile_payload.keys() if not str(name).startswith("_")]
    if not datasets:
        raise RuntimeError(f"No datasets found for profile {args.profile!r}.")
    output: Dict[str, Any] = {
        "schema_version": "utility-sensitivity-audit-v1",
        "legacy_schema_alias": "utility-causal-diagnostics-v1",
        "purpose": "Official Utility probe sensitivity audit for separating probe validity, curation benefit, and strict counterfactual benefit.",
        "profile": args.profile,
        "baseline_roles": {
            "baseline_stageA_random": "curation_benefit_baseline",
            "baseline_multi_matched_stageA_random": "strict_counterfactual_certification_baseline",
            "other_matched_or_full_random_baselines": "diagnostic_stress_tests",
        },
        "protocol": {
            "baseline": CANONICAL_BASELINE,
            "baseline_policy": "common_stageA_baseline_disjoint_from_all_sensitivity_arms",
            "sensitivity_design": "two_stage_destructive_plus_token_inventory_stress",
            "destructive_negative_control": "corrupted_negative_control",
            "token_inventory_stress_control": "token_shuffle_negative_control",
            "train_token_budget": int(args.train_token_budget),
            "eval_token_budget": int(args.eval_token_budget),
            "max_train_steps": int(args.max_train_steps),
            "train_epochs": float(args.train_epochs),
            "holdout_bucket": int(args.holdout_bucket),
            "seed": int(args.seed),
            "arm_pool_size": int(args.arm_pool_size),
            "corruption_mode": str(args.corruption_mode),
        },
        "datasets": {},
    }
    conn = sqlite3.connect(str(INDEX_DB_PATH))
    try:
        for dataset in datasets:
            print(f"[14] dataset start: {dataset}", flush=True)
            scored = _load_scored(str(dataset))
            selected_uids = _load_selected_uids(run_summary, args.profile, str(dataset))
            stage_a_records = [record for record in scored if _stage_a_pass(record)]
            stage_a_uids = {str(record.get("chunk_uid") or "") for record in stage_a_records}
            effective_arm_pool_size = _effective_arm_pool_size(
                stage_a_count=len(stage_a_records),
                arm_count=len(args.arms),
                requested=int(args.arm_pool_size),
            )
            records_by_arm: Dict[str, List[Dict[str, Any]]] = {}
            all_arm_uids: set[str] = set()
            for arm in args.arms:
                records = _arm_records(
                    dataset=str(dataset),
                    scored_records=scored,
                    selected_uids=selected_uids,
                    arm=str(arm),
                    arm_pool_size=int(effective_arm_pool_size),
                    seed=int(args.seed),
                )
                records_by_arm[str(arm)] = records
                all_arm_uids.update(str(record.get("chunk_uid") or "") for record in records if str(record.get("chunk_uid") or ""))
            common_baseline_allowed_uids = set(stage_a_uids) - all_arm_uids
            if not common_baseline_allowed_uids:
                raise RuntimeError(
                    f"{dataset}: common sensitivity baseline pool is empty after excluding all diagnostic arms."
                )
            arm_results = []
            for arm in args.arms:
                print(f"[14] arm start: dataset={dataset} arm={arm}", flush=True)
                records = records_by_arm[str(arm)]
                result = _score_arm(
                    conn=conn,
                    dataset=str(dataset),
                    arm=str(arm),
                    arm_records=records,
                    stage_a_uids=stage_a_uids,
                    common_baseline_allowed_uids=common_baseline_allowed_uids,
                    args=args,
                )
                arm_results.append(result)
                print(
                    f"[14] arm done: dataset={dataset} arm={arm} "
                    f"gain={result.get('small_lm_probe_gain_score')} delta={result.get('delta_nll')} "
                    f"causal={result.get('causal_failure_mode')}",
                    flush=True,
                )
            current_failure = _current_run_failure_decomposition(run_summary, args.profile, str(dataset))
            probe_sensitivity = _probe_sensitivity_order(arm_results)
            output["datasets"][str(dataset)] = {
                "stage_a_records": int(len(stage_a_records)),
                "requested_arm_pool_size": int(args.arm_pool_size),
                "effective_arm_pool_size": int(effective_arm_pool_size),
                "common_baseline_pool_records": int(len(common_baseline_allowed_uids)),
                "common_baseline_uid_fingerprint": hashlib.sha1(
                    "\n".join(sorted(common_baseline_allowed_uids)).encode()
                ).hexdigest(),
                "current_run_failure_decomposition": current_failure,
                "arm_results": arm_results,
                "probe_sensitivity": probe_sensitivity,
                "root_cause_decision": _root_cause_decision(probe_sensitivity, current_failure),
                "selector_tuning_allowed": False,
                "selector_policy_action": "hold",
                "utility_scope": "Stage C diagnostic only; never selector objective",
            }
            print(f"[14] dataset done: {dataset}", flush=True)
    finally:
        conn.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output == DEFAULT_OUTPUT_PATH:
        LEGACY_OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.no_update_artifacts:
        print("[14] skipped run_summary/run_manifest update (--no-update-artifacts)", flush=True)
    else:
        _update_run_artifacts_with_audit(output)
    print(f"[14] utility sensitivity audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
