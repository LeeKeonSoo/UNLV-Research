#!/usr/bin/env python3
"""Build a review-only sample from a frozen source-role diagnostic model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


JsonMap = dict[str, Any]
RAW_TIER = "raw_like"
MAX_EXCERPT_CHARS = 800


def _stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _raw_like(row: JsonMap) -> bool:
    partition = row.get("partition")
    return isinstance(partition, dict) and partition.get("source_tier") == RAW_TIER


def _text(row: JsonMap) -> str:
    return str(row.get("text") or "")


def _source_summary(row: JsonMap) -> JsonMap:
    source = row.get("source")
    partition = row.get("partition")
    provenance = row.get("provenance")
    source_map = source if isinstance(source, dict) else {}
    partition_map = partition if isinstance(partition, dict) else {}
    provenance_map = provenance if isinstance(provenance, dict) else {}
    return {
        "dataset": source_map.get("dataset") or partition_map.get("source_dataset"),
        "repository": source_map.get("repository") or partition_map.get("repository_identity") or provenance_map.get("source_name"),
        "path": source_map.get("path") or partition_map.get("path"),
    }


def _review_record(row: JsonMap, score: float) -> JsonMap:
    metadata = _source_summary(row)
    return {
        "record_id": str(row.get("record_id") or ""),
        "origin_record_id": str(row.get("origin_record_id") or row.get("record_id") or ""),
        "reference_distribution_score": round(score, 6),
        "dataset": metadata.get("dataset"),
        "repository": metadata.get("repository"),
        "path": metadata.get("path"),
        "text_excerpt": _text(row)[:MAX_EXCERPT_CHARS],
        "label_status": "unlabeled",
        "allowed_dispositions": ["retain", "unsupported_scope", "false_positive", "needs_more_evidence"],
    }


def build_review_sample(
    reference_train: Iterable[JsonMap], candidate_rows: Iterable[JsonMap], *, split_salt: str, sample_size: int
) -> JsonMap:
    """Rank raw-like records for review without emitting selection decisions."""
    references = list(reference_train)
    raw_candidates = [row for row in candidate_rows if _raw_like(row)]
    raw_train = sorted(raw_candidates, key=lambda row: _stable_key(str(row.get("record_id") or ""), split_salt))[: len(references)]
    if not references or len(raw_train) != len(references):
        raise RuntimeError("Review sample needs balanced reference and raw-like training examples.")
    if sample_size <= 0:
        raise RuntimeError("Review sample size must be positive.")

    raw_train_ids = {str(row.get("record_id") or "") for row in raw_train}
    review_candidates = [row for row in raw_candidates if str(row.get("record_id") or "") not in raw_train_ids]
    if not review_candidates:
        raise RuntimeError("Review sample needs at least one raw-like record outside the training sample.")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, sublinear_tf=True)
    classifier = LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)
    train_text = [*[_text(row) for row in references], *[_text(row) for row in raw_train]]
    train_target = [1] * len(references) + [0] * len(raw_train)
    classifier.fit(vectorizer.fit_transform(train_text), train_target)
    scores = classifier.predict_proba(vectorizer.transform([_text(row) for row in review_candidates]))[:, 1]
    ranked = sorted(zip(review_candidates, scores, strict=True), key=lambda item: (-item[1], str(item[0].get("record_id") or "")))
    review_records = [_review_record(row, float(score)) for row, score in ranked[:sample_size]]
    return {
        "schema_version": "reference-distribution-review-sample-v1",
        "status": "review_sample_ready_labels_required_not_a_selection_policy",
        "review_purpose": "False-positive and scope audit of a source-role diagnostic. No record is selected or removed.",
        "summary": {
            "reference_training_records": len(references),
            "raw_like_training_records": len(raw_train),
            "eligible_review_candidates": len(review_candidates),
            "review_records": len(review_records),
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
            "candidate_scores_materialized": True,
            "selection_decisions_emitted": False,
            "data_removed": False,
        },
        "claim_boundary": "Scores are review ordering for a declared source-role diagnostic only. Labels and an independent scope audit are required before any selector implementation.",
        "review_records": review_records,
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a review-only source-role diagnostic sample.")
    parser.add_argument("--reference-train", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--split-salt", default="calibrated-selector-v1")
    args = parser.parse_args()
    report = build_review_sample(
        _read_jsonl(args.reference_train),
        _read_jsonl(args.candidate_input),
        split_salt=args.split_salt,
        sample_size=args.sample_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
