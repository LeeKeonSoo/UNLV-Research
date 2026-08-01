#!/usr/bin/env python3
"""Diagnose why scaled SLM pilot behavior changes at full budget."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from data_eval_common import OUTPUT_DIR, iter_jsonl_records_resilient, load_json, save_json
from policy.subsets import SCORING_MANIFEST_PATH


DEFAULT_EXPERIMENT_DIR = OUTPUT_DIR / "slm_update_experiments" / "fineweb_edu_canonical_slm_update_v1"
DEFAULT_PLAN = Path(__file__).resolve().parent / "configs" / "slm_update_certification_plan_qwen25_0p5b_fineweb.json"
PRIMARY_ARMS = ("curated_equal_budget", "stageA_random_equal_budget")
SUPPORTING_ARMS = ("raw_random_equal_budget",)
SLICE_SEQUENCE_LENGTH = 1024
PILOT_SEQUENCES = 1024


def _uid(record: Dict[str, Any]) -> str:
    return str(record.get("chunk_uid") or record.get("id") or record.get("doc_id") or "")


def _score(core: Dict[str, Any], metric: str) -> float | None:
    value = ((core.get(metric) or {}).get("score"))
    return float(value) if isinstance(value, (int, float)) else None


def _details(core: Dict[str, Any], metric: str) -> Dict[str, Any]:
    payload = core.get(metric) if isinstance(core.get(metric), dict) else {}
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    return details


def _entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log(count / total) for count in counter.values() if count > 0))


def _weighted_mean(pairs: Iterable[Tuple[float | None, int]]) -> float | None:
    total_weight = 0
    total = 0.0
    for value, weight in pairs:
        if value is None or weight <= 0:
            continue
        total += float(value) * int(weight)
        total_weight += int(weight)
    return total / total_weight if total_weight else None


def _load_scored_by_uid(dataset: str) -> Dict[str, Dict[str, Any]]:
    manifest = load_json(SCORING_MANIFEST_PATH)
    path = Path(str(((manifest.get("datasets") or {}).get(dataset) or {}).get("path") or ""))
    if not path.exists():
        raise FileNotFoundError(f"Missing scored path for {dataset}: {path}")
    return {_uid(record): record for record in iter_jsonl_records_resilient(path)}


def _load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _record_token_count(record: Dict[str, Any], tokenizer: Any) -> int:
    text = str(record.get("text") or "")
    if not text.strip():
        return 0
    count = len(tokenizer(text, add_special_tokens=False).input_ids)
    return count + (1 if getattr(tokenizer, "eos_token_id", None) is not None else 0)


def _slice_records(path: Path, tokenizer: Any, max_tokens: int) -> List[Tuple[Dict[str, Any], int, int]]:
    rows: List[Tuple[Dict[str, Any], int, int]] = []
    consumed = 0
    for record in iter_jsonl_records_resilient(path):
        token_count = _record_token_count(record, tokenizer)
        if token_count <= 0:
            continue
        remaining = max_tokens - consumed
        if remaining <= 0:
            break
        used = min(token_count, remaining)
        rows.append((record, token_count, used))
        consumed += used
        if consumed >= max_tokens:
            break
    return rows


def _all_records(path: Path, tokenizer: Any) -> List[Tuple[Dict[str, Any], int, int]]:
    rows: List[Tuple[Dict[str, Any], int, int]] = []
    for record in iter_jsonl_records_resilient(path):
        token_count = _record_token_count(record, tokenizer)
        if token_count <= 0:
            continue
        rows.append((record, token_count, token_count))
    return rows


def _summary(rows: List[Tuple[Dict[str, Any], int, int]], scored: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source_tokens: Counter[str] = Counter()
    style_tokens: Counter[str] = Counter()
    cluster_tokens: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    qualities: List[Tuple[float | None, int]] = []
    redundancy: List[Tuple[float | None, int]] = []
    repeat_pressure: List[Tuple[float | None, int]] = []
    useful_recurrence: List[Tuple[float | None, int]] = []
    validity_soft: List[Tuple[float | None, int]] = []
    word_counts: List[Tuple[float | None, int]] = []
    token_counts: List[Tuple[float | None, int]] = []
    missing_scored = 0
    total_tokens = 0
    total_record_tokens = 0
    total_words = 0
    for record, record_tokens, used_tokens in rows:
        uid = _uid(record)
        scored_record = scored.get(uid)
        if not scored_record:
            missing_scored += 1
            core: Dict[str, Any] = {}
            diagnostics: Dict[str, Any] = {}
        else:
            core = scored_record.get("core_metrics") if isinstance(scored_record.get("core_metrics"), dict) else {}
            diagnostics = scored_record.get("diagnostics") if isinstance(scored_record.get("diagnostics"), dict) else {}
        source = str(record.get("source") or "unknown")
        source_tokens[source] += used_tokens
        structural = _details(core, "structural_validity_gate")
        style = str(structural.get("style_bucket") or "unknown")
        style_tokens[style] += used_tokens
        provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
        metadata = provenance.get("metadata") if isinstance(provenance.get("metadata"), dict) else {}
        source_metadata = metadata.get("source_metadata") if isinstance(metadata.get("source_metadata"), dict) else {}
        domain = str(metadata.get("domain") or source_metadata.get("url") or "unknown").split("/")[2] if "/" in str(metadata.get("domain") or source_metadata.get("url") or "") else str(metadata.get("domain") or "unknown")
        domain_tokens[domain] += used_tokens
        cluster_tokens[str(diagnostics.get("cluster_id", "unknown"))] += used_tokens
        q = _score(core, "reference_quality_score")
        red = _score(core, "shingle_near_duplicate_risk_score")
        red_details = _details(core, "shingle_near_duplicate_risk_score")
        val_details = _details(core, "structural_validity_gate")
        qualities.append((q, used_tokens))
        redundancy.append((red, used_tokens))
        repeat_pressure.append((float(red_details.get("intra_chunk_repeat_pressure")) if isinstance(red_details.get("intra_chunk_repeat_pressure"), (int, float)) else None, used_tokens))
        useful_recurrence.append((float(red_details.get("useful_recurrence_score")) if isinstance(red_details.get("useful_recurrence_score"), (int, float)) else None, used_tokens))
        validity_soft.append((float(val_details.get("soft_score")) if isinstance(val_details.get("soft_score"), (int, float)) else None, used_tokens))
        word_count = int(record.get("word_count") or 0)
        word_counts.append((float(word_count), used_tokens))
        token_counts.append((float(record_tokens), used_tokens))
        total_tokens += used_tokens
        total_record_tokens += record_tokens
        total_words += word_count
    top_source_tokens = source_tokens.most_common(10)
    top_cluster_tokens = cluster_tokens.most_common(10)
    top_domain_tokens = domain_tokens.most_common(10)
    top_source_share = (top_source_tokens[0][1] / total_tokens) if top_source_tokens and total_tokens else 0.0
    top_cluster_share = (top_cluster_tokens[0][1] / total_tokens) if top_cluster_tokens and total_tokens else 0.0
    return {
        "records_touched": len(rows),
        "tokens_used": int(total_tokens),
        "record_tokens_touched": int(total_record_tokens),
        "word_count_touched": int(total_words),
        "missing_scored_records": int(missing_scored),
        "mean_record_tokens_weighted": _weighted_mean(token_counts),
        "mean_record_words_weighted": _weighted_mean(word_counts),
        "quality_token_weighted_mean": _weighted_mean(qualities),
        "redundancy_risk_token_weighted_mean": _weighted_mean(redundancy),
        "repeat_pressure_token_weighted_mean": _weighted_mean(repeat_pressure),
        "useful_recurrence_token_weighted_mean": _weighted_mean(useful_recurrence),
        "validity_soft_token_weighted_mean": _weighted_mean(validity_soft),
        "unique_sources": len(source_tokens),
        "source_entropy": _entropy(source_tokens),
        "effective_source_count": math.exp(_entropy(source_tokens)) if source_tokens else 0.0,
        "top_source_token_share": top_source_share,
        "top_sources": dict(top_source_tokens),
        "style_token_distribution": dict(style_tokens.most_common()),
        "unique_domains": len(domain_tokens),
        "domain_entropy": _entropy(domain_tokens),
        "effective_domain_count": math.exp(_entropy(domain_tokens)) if domain_tokens else 0.0,
        "top_domains": dict(top_domain_tokens),
        "unique_clusters": len(cluster_tokens),
        "cluster_entropy": _entropy(cluster_tokens),
        "effective_cluster_count": math.exp(_entropy(cluster_tokens)) if cluster_tokens else 0.0,
        "top_cluster_token_share": top_cluster_share,
        "top_clusters": dict(top_cluster_tokens),
    }


def _delta(left: Dict[str, Any], right: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in fields:
        a = left.get(field)
        b = right.get(field)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out[field] = float(a) - float(b)
    return out


def build_report(experiment_dir: Path, plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    manifest = load_json(experiment_dir / "manifest.json")
    frozen = load_json(experiment_dir / "frozen_training_plan.json")
    dataset = str(manifest.get("dataset") or "")
    scored = _load_scored_by_uid(dataset)
    model_id = str((frozen.get("target_model") or {}).get("model_id") or plan.get("model") or "")
    tokenizer = _load_tokenizer(model_id)
    arms = manifest.get("arms") if isinstance(manifest.get("arms"), dict) else {}
    pilot_tokens = PILOT_SEQUENCES * SLICE_SEQUENCE_LENGTH
    full_tokens = int(((frozen.get("token_budget") or {}).get("all_equal_budget_arms_matched_token_budget") or 0))
    slices: Dict[str, Dict[str, Any]] = {}
    for arm in list(PRIMARY_ARMS) + list(SUPPORTING_ARMS):
        path = Path(str((arms.get(arm) or {}).get("path") or ""))
        if not path.exists():
            continue
        slices[arm] = {
            "pilot_1024": _summary(_slice_records(path, tokenizer, pilot_tokens), scored),
            "full_budget": _summary(_slice_records(path, tokenizer, full_tokens), scored),
        }
    eval_holdout_manifest = load_json(experiment_dir / "eval_holdout_manifest.json") if (experiment_dir / "eval_holdout_manifest.json").exists() else {}
    eval_path = Path(str(((eval_holdout_manifest.get("paths") or {}).get("eval_jsonl") or experiment_dir / "heldout_stageA_eval.jsonl")))
    eval_summary = _summary(_all_records(eval_path, tokenizer), scored) if eval_path.exists() else {}
    diagnostic_fields = [
        "records_touched",
        "mean_record_tokens_weighted",
        "quality_token_weighted_mean",
        "redundancy_risk_token_weighted_mean",
        "repeat_pressure_token_weighted_mean",
        "useful_recurrence_token_weighted_mean",
        "validity_soft_token_weighted_mean",
        "effective_source_count",
        "top_source_token_share",
        "effective_domain_count",
        "effective_cluster_count",
        "top_cluster_token_share",
    ]
    comparisons = {
        "curated_minus_stageA_random_pilot_1024": _delta(
            slices["curated_equal_budget"]["pilot_1024"],
            slices["stageA_random_equal_budget"]["pilot_1024"],
            diagnostic_fields,
        ),
        "curated_minus_stageA_random_full_budget": _delta(
            slices["curated_equal_budget"]["full_budget"],
            slices["stageA_random_equal_budget"]["full_budget"],
            diagnostic_fields,
        ),
        "curated_full_minus_pilot": _delta(
            slices["curated_equal_budget"]["full_budget"],
            slices["curated_equal_budget"]["pilot_1024"],
            diagnostic_fields,
        ),
        "stageA_random_full_minus_pilot": _delta(
            slices["stageA_random_equal_budget"]["full_budget"],
            slices["stageA_random_equal_budget"]["pilot_1024"],
            diagnostic_fields,
        ),
    }
    if eval_summary:
        comparisons.update(
            {
                "eval_minus_curated_full_budget": _delta(
                    eval_summary,
                    slices["curated_equal_budget"]["full_budget"],
                    diagnostic_fields,
                ),
                "eval_minus_stageA_random_full_budget": _delta(
                    eval_summary,
                    slices["stageA_random_equal_budget"]["full_budget"],
                    diagnostic_fields,
                ),
            }
        )
    train_eval = {
        "scaled_pilot_report": load_json(experiment_dir / "pilot_1024_lr1e5_scaled_report.json")
        if (experiment_dir / "pilot_1024_lr1e5_scaled_report.json").exists()
        else {},
        "certification_report": load_json(experiment_dir / "cert_lr1e5_full_certification_report.json")
        if (experiment_dir / "cert_lr1e5_full_certification_report.json").exists()
        else {},
    }
    report = {
        "schema_version": "slm-full-budget-shift-diagnostic-v1",
        "experiment_dir": str(experiment_dir),
        "plan_path": str(plan_path),
        "scope": "diagnostic_only_not_selector_objective",
        "utility_scope": "Stage C validation only; never selector objective",
        "question": "Why did curated beat Stage-A random in the 1024-sequence pilot but lose in the first full-budget certification seed?",
        "slice_definitions": {
            "pilot_1024_tokens": int(pilot_tokens),
            "full_budget_tokens": int(full_tokens),
            "sequence_length": SLICE_SEQUENCE_LENGTH,
        },
        "arm_slices": slices,
        "eval_holdout_summary": eval_summary,
        "comparisons": comparisons,
        "train_eval_context": train_eval,
        "interpretation_hints": [
            "If curated full budget is less diverse than its pilot prefix, full exposure may amplify narrowness.",
            "If Stage-A random has much lower train loss but similar or better eval NLL, selected data may be harder/narrower rather than more generally useful.",
            "Large source/domain/cluster concentration differences point to coverage loss under full exposure.",
            "These diagnostics must not be used directly as Stage-B Utility optimization targets.",
        ],
    }
    save_json(experiment_dir / "slm_full_budget_shift_diagnostic.json", report)
    md_lines = [
        "# SLM Full-Budget Shift Diagnostic",
        "",
        "Scope: diagnostic only; not selector objective.",
        "",
        "## Key Comparisons",
        "",
    ]
    for name, payload in comparisons.items():
        md_lines.append(f"### `{name}`")
        md_lines.append("")
        md_lines.append("| Field | Delta |")
        md_lines.append("| --- | ---: |")
        for field, value in payload.items():
            md_lines.append(f"| `{field}` | {float(value):.6f} |")
        md_lines.append("")
    md_lines.extend([
        "## Reminder",
        "",
        "These diagnostics explain the Stage-C target-SLM result. Do not feed target-SLM outcomes into Stage-B selection.",
        "",
    ])
    (experiment_dir / "slm_full_budget_shift_diagnostic.md").write_text("\n".join(md_lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose SLM full-budget shift.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    report = build_report(args.experiment_dir, args.plan)
    print(
        {
            "scope": report["scope"],
            "comparisons": report["comparisons"],
            "output": str(args.experiment_dir / "slm_full_budget_shift_diagnostic.md"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
