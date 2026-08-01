#!/usr/bin/env python3
"""Compare selected, usable-not-selected, and Stage-A-rejected OpenWebText2 slices."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, SCORED_DIR, iter_jsonl_records_resilient, load_json, save_json


DEFAULT_DATASET = "openwebtext2_subset"
DEFAULT_PROFILE = "paper_release_certification"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "openwebtext2_slice_diagnostic.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "openwebtext2_slice_diagnostic.md"
SELECTOR_AUDIT_PATH = OUTPUT_DIR / "validation" / "selector_baseline_audit.json"

FEATURES = (
    "quality",
    "redundancy_risk",
    "structural_validity",
    "word_count",
    "lexical_diversity",
    "repeat_pressure",
    "useful_recurrence",
    "predictive_quality_support",
    "predictive_utility_proxy",
)
TEXT_PATTERNS = {
    "seo_or_marketing": re.compile(r"\b(?:SEO|meta description|keywords?|sign me up|newsletter|advertisement)\b", re.I),
    "news_or_current_event": re.compile(r"\b(?:reported|according to|spokesman|officials?|press conference|news)\b", re.I),
    "instructional": re.compile(r"\b(?:how to|step \d+|tutorial|instructions?|guide)\b", re.I),
    "code_or_technical": re.compile(r"\b(?:software|developer|code|API|security|database|algorithm)\b", re.I),
    "mojibake_marker": re.compile(r"(?:�|Ã|Â|â€™|â€œ|\?\?)"),
}


def _metric(record: Dict[str, Any], group: str, name: str, detail: str | None = None) -> float:
    payload = ((record.get(group) or {}).get(name) or {})
    value = ((payload.get("details") or {}).get(detail)) if detail else payload.get("score")
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _features(record: Dict[str, Any]) -> Dict[str, float]:
    validity_details = (((record.get("core_metrics") or {}).get("structural_validity_gate") or {}).get("details") or {})
    redundancy_details = (((record.get("core_metrics") or {}).get("shingle_near_duplicate_risk_score") or {}).get("details") or {})
    predictive_details = (((record.get("diagnostic_metrics") or {}).get("predictive_utility_proxy") or {}).get("details") or {})
    return {
        "quality": _metric(record, "core_metrics", "reference_quality_score"),
        "redundancy_risk": _metric(record, "core_metrics", "shingle_near_duplicate_risk_score"),
        "structural_validity": _metric(record, "diagnostic_metrics", "structural_validity_score"),
        "word_count": float(record.get("word_count") or 0),
        "lexical_diversity": float(validity_details.get("lexical_diversity") or 0.0),
        "repeat_pressure": float(redundancy_details.get("intra_chunk_repeat_pressure") or 0.0),
        "useful_recurrence": float(redundancy_details.get("useful_recurrence_score") or 0.0),
        "predictive_quality_support": float(predictive_details.get("quality_support") or 0.0),
        "predictive_utility_proxy": _metric(record, "diagnostic_metrics", "predictive_utility_proxy"),
    }


def _stage_a_pass(record: Dict[str, Any]) -> bool:
    core = record.get("core_metrics") or {}
    return bool(
        ((core.get("structural_validity_gate") or {}).get("score") or 0) >= 1
        and ((core.get("exact_duplicate_indicator") or {}).get("score") or 0) <= 0
        and ((core.get("shingle_near_duplicate_indicator") or {}).get("score") or 0) <= 0
    )


def _new_slice() -> Dict[str, Any]:
    return {
        "records": 0,
        "feature_sums": Counter(),
        "style_counts": Counter(),
        "text_pattern_counts": Counter(),
        "examples": [],
    }


def _add_record(slice_payload: Dict[str, Any], record: Dict[str, Any], *, example_limit: int) -> None:
    slice_payload["records"] += 1
    for name, value in _features(record).items():
        slice_payload["feature_sums"][name] += value
    validity_details = (((record.get("core_metrics") or {}).get("structural_validity_gate") or {}).get("details") or {})
    slice_payload["style_counts"][str(validity_details.get("style_bucket") or "unknown")] += 1
    preview = str(((record.get("provenance") or {}).get("text_preview") or ""))
    for name, pattern in TEXT_PATTERNS.items():
        if pattern.search(preview):
            slice_payload["text_pattern_counts"][name] += 1
    if len(slice_payload["examples"]) < example_limit:
        digest = hashlib.sha256(str(record.get("chunk_uid") or "").encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % 97 < 3:
            slice_payload["examples"].append(
                {
                    "chunk_uid": record.get("chunk_uid"),
                    "source": record.get("source"),
                    "quality": round(_features(record)["quality"], 6),
                    "preview": preview[:280],
                }
            )


def _finalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    count = int(payload["records"])
    return {
        "records": count,
        "feature_means": {
            name: round(float(payload["feature_sums"].get(name, 0.0)) / max(count, 1), 6)
            for name in FEATURES
        },
        "style_distribution": {
            name: {"count": value, "share": round(value / max(count, 1), 6)}
            for name, value in payload["style_counts"].most_common()
        },
        "text_pattern_rates": {
            name: round(float(payload["text_pattern_counts"].get(name, 0)) / max(count, 1), 6)
            for name in TEXT_PATTERNS
        },
        "examples": payload["examples"],
    }


def _delta(selected: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    return {
        name: round(
            float((selected.get("feature_means") or {}).get(name) or 0.0)
            - float((baseline.get("feature_means") or {}).get(name) or 0.0),
            6,
        )
        for name in FEATURES
    }


def build_report(
    *,
    dataset: str,
    profile: str,
    scored_path: Path,
    selected_path: Path,
    example_limit: int,
) -> Dict[str, Any]:
    selected_uids = {
        str(record.get("chunk_uid"))
        for record in iter_jsonl_records_resilient(selected_path)
        if record.get("chunk_uid")
    }
    slices = {
        "selected": _new_slice(),
        "stage_a_usable_not_selected": _new_slice(),
        "stage_a_rejected": _new_slice(),
    }
    for record in iter_jsonl_records_resilient(scored_path):
        uid = str(record.get("chunk_uid") or "")
        if uid in selected_uids:
            bucket = "selected"
        elif _stage_a_pass(record):
            bucket = "stage_a_usable_not_selected"
        else:
            bucket = "stage_a_rejected"
        _add_record(slices[bucket], record, example_limit=example_limit)

    finalized = {name: _finalize(payload) for name, payload in slices.items()}
    usable = finalized["stage_a_usable_not_selected"]
    selected = finalized["selected"]
    deltas = _delta(selected, usable)
    selected_styles = selected["style_distribution"]
    usable_styles = usable["style_distribution"]
    technical_shift = round(
        float((selected_styles.get("technical_reference") or {}).get("share") or 0.0)
        - float((usable_styles.get("technical_reference") or {}).get("share") or 0.0),
        6,
    )
    hypotheses = [
        {
            "id": "H-OWT2-01",
            "hypothesis": "Stage B may remove repetition/length signals that the small-LM probe learns easily.",
            "status": "supported_as_diagnostic_candidate" if deltas["repeat_pressure"] < -0.05 else "not_supported_by_slice_means",
            "evidence": {
                "selected_minus_usable_repeat_pressure": deltas["repeat_pressure"],
                "selected_minus_usable_word_count": deltas["word_count"],
            },
            "next_test": "Run equal-token target-SLM ablation comparing canonical selection with a recurrence/length-controlled selection.",
        },
        {
            "id": "H-OWT2-02",
            "hypothesis": "The selector may concentrate a style favored by the reference-quality model.",
            "status": "supported_as_diagnostic_candidate" if abs(technical_shift) >= 0.1 else "not_supported_by_slice_means",
            "evidence": {"selected_minus_usable_technical_reference_share": technical_shift},
            "next_test": "Audit quality labels and Utility by style slice; do not add Utility to the selector objective.",
        },
        {
            "id": "H-OWT2-03",
            "hypothesis": "Raw-web extraction artifacts remain after Stage A and may weaken both selected and baseline training.",
            "status": (
                "supported_as_diagnostic_candidate"
                if selected["text_pattern_rates"]["mojibake_marker"] > 0.001
                else "not_supported_by_preview_rate"
            ),
            "evidence": {
                "selected_mojibake_preview_rate": selected["text_pattern_rates"]["mojibake_marker"],
                "usable_mojibake_preview_rate": usable["text_pattern_rates"]["mojibake_marker"],
            },
            "next_test": "Re-ingest a raw-web sample through Stage 0 and compare before/after Stage-C evidence.",
        },
    ]
    return {
        "schema_version": "openwebtext2-slice-diagnostic-v1",
        "dataset": dataset,
        "profile": profile,
        "purpose": "Diagnose why Core-feature gains may not transfer to Utility without using Utility as a selector objective.",
        "scope": "diagnostic only",
        "slices": finalized,
        "selected_minus_stage_a_usable_not_selected": deltas,
        "style_shift": {"technical_reference_share_delta": technical_shift},
        "hypotheses": hypotheses,
        "selector_action": "hold",
    }


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# OpenWebText2 Slice Diagnostic",
        "",
        f"- Dataset: `{report.get('dataset')}`",
        f"- Profile: `{report.get('profile')}`",
        "- Scope: diagnostic only; Utility remains Stage C and is not a selector objective.",
        "",
        "## Slice Summary",
        "",
        "| Slice | Records | Quality | Redundancy risk | Repeat pressure | Word count | Predictive proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in (report.get("slices") or {}).items():
        means = payload.get("feature_means") or {}
        lines.append(
            f"| {name} | {payload.get('records')} | {means.get('quality')} | {means.get('redundancy_risk')} | "
            f"{means.get('repeat_pressure')} | {means.get('word_count')} | {means.get('predictive_utility_proxy')} |"
        )
    lines.extend(["", "## Diagnostic Hypotheses", ""])
    for hypothesis in report.get("hypotheses") or []:
        lines.extend(
            [
                f"### {hypothesis.get('id')}",
                "",
                f"- Hypothesis: {hypothesis.get('hypothesis')}",
                f"- Status: `{hypothesis.get('status')}`",
                f"- Evidence: `{hypothesis.get('evidence')}`",
                f"- Next test: {hypothesis.get('next_test')}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenWebText2 selected/rejected slice diagnostic.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--scored-path", type=Path)
    parser.add_argument("--selected-path", type=Path)
    parser.add_argument("--example-limit", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    scored_path = args.scored_path or (SCORED_DIR / f"{args.dataset}.jsonl")
    selected_path = args.selected_path or (OUTPUT_DIR / "subsets" / args.profile / f"{args.dataset}.jsonl")
    report = build_report(
        dataset=args.dataset,
        profile=args.profile,
        scored_path=scored_path,
        selected_path=selected_path,
        example_limit=args.example_limit,
    )
    save_json(args.output, report)
    write_markdown(report, args.md_output)
    print(f"[31] OpenWebText2 slice diagnostic: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
