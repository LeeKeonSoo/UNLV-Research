#!/usr/bin/env python3
"""Measure a frozen source-role proxy without selecting or removing data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


JsonMap = dict[str, Any]
RAW_TIER = "raw_like"


def _stable_key(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def _raw_like(row: JsonMap) -> bool:
    partition = row.get("partition")
    return isinstance(partition, dict) and partition.get("source_tier") == RAW_TIER


def _text(row: JsonMap) -> str:
    return str(row.get("text") or "")


def _calibration_target(row: JsonMap) -> int:
    label = row.get("source_role_label")
    if label == "reference_distribution_member":
        return 1
    if label == "raw_like_nonmember":
        return 0
    raise RuntimeError("Calibration row has an unsupported source-role label.")


def run_probe(
    reference_train: Iterable[JsonMap], candidate_rows: Iterable[JsonMap], calibration_rows: Iterable[JsonMap], *, split_salt: str
) -> JsonMap:
    """Fit and evaluate a source-role probe on repository-disjoint frozen data."""
    references = list(reference_train)
    calibration = list(calibration_rows)
    calibration_raw_ids = {
        str(row.get("origin_record_id") or "")
        for row in calibration
        if row.get("source_role_label") == "raw_like_nonmember"
    }
    raw_candidates = [
        row
        for row in candidate_rows
        if _raw_like(row) and str(row.get("record_id") or "") not in calibration_raw_ids
    ]
    raw_train = sorted(raw_candidates, key=lambda row: _stable_key(str(row.get("record_id") or ""), split_salt))[: len(references)]
    if not references or len(raw_train) != len(references):
        raise RuntimeError("Probe needs balanced reference and raw-like training examples.")
    if len({_calibration_target(row) for row in calibration}) != 2:
        raise RuntimeError("Held-out calibration needs both declared source-role labels.")

    train_text = [*[_text(row) for row in references], *[_text(row) for row in raw_train]]
    train_target = [1] * len(references) + [0] * len(raw_train)
    calibration_text = [_text(row) for row in calibration]
    calibration_target = [_calibration_target(row) for row in calibration]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000, sublinear_tf=True)
    classifier = LogisticRegression(class_weight="balanced", max_iter=500, random_state=0)
    train_matrix = vectorizer.fit_transform(train_text)
    classifier.fit(train_matrix, train_target)
    probabilities = classifier.predict_proba(vectorizer.transform(calibration_text))[:, 1]
    return {
        "schema_version": "reference-distribution-probe-v1",
        "status": "diagnostic_probe_complete_not_a_selection_policy",
        "selection_hypothesis": "reference_distribution_membership_for_declared_code_scope",
        "model": {
            "kind": "tfidf_character_ngram_logistic_regression",
            "ngram_range": [3, 5],
            "max_features": 50000,
            "random_state": 0,
        },
        "training": {
            "reference_positive_records": len(references),
            "raw_like_negative_records": len(raw_train),
        },
        "held_out_calibration": {
            "records": len(calibration),
            "positive_records": sum(calibration_target),
            "negative_records": len(calibration_target) - sum(calibration_target),
            "roc_auc": round(float(roc_auc_score(calibration_target, probabilities)), 6),
            "average_precision": round(float(average_precision_score(calibration_target, probabilities)), 6),
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "target_token_fraction_read": False,
            "candidate_scores_materialized": False,
            "data_removed": False,
        },
        "claim_boundary": "This measures only source-role classification on a frozen repository-disjoint split. It does not measure intrinsic Quality, training Utility, or authorize Stage C selection.",
    }


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a non-selecting reference-distribution diagnostic probe.")
    parser.add_argument("--reference-train", type=Path, required=True)
    parser.add_argument("--candidate-input", type=Path, required=True)
    parser.add_argument("--held-out-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-salt", default="calibrated-selector-v1")
    args = parser.parse_args()
    report = run_probe(
        _read_jsonl(args.reference_train),
        _read_jsonl(args.candidate_input),
        _read_jsonl(args.held_out_calibration),
        split_salt=args.split_salt,
    )
    report["artifacts"] = {
        "reference_train": {"path": str(args.reference_train), "sha256": _sha256(args.reference_train)},
        "candidate_input": {"path": str(args.candidate_input), "sha256": _sha256(args.candidate_input)},
        "held_out_calibration": {"path": str(args.held_out_calibration), "sha256": _sha256(args.held_out_calibration)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "held_out_calibration": report["held_out_calibration"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
