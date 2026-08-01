#!/usr/bin/env python3
"""Build a labeled Redundancy calibration and saturation benchmark."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_chunks import _hard_overlap, token_shingles
from ingestion.code_fingerprints import derived_fingerprints, simhash_hamming_distance
from ingestion.code_selection import score_stage_b


DEFAULT_FIXTURES = Path("validation") / "fixtures" / "redundancy_validity_benchmark_cases.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_validity_benchmark_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_validity_benchmark_report.md"
SIMHASH_THRESHOLDS = [0, 1, 2, 3, 5, 8, 10, 16, 18, 20, 24, 32]
JACCARD_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
CONTAINMENT_THRESHOLDS = [0.75, 0.80, 0.88, 0.95]


def _pair_evidence(case: Dict[str, Any]) -> Dict[str, Any]:
    left = str(case["left"])
    right = str(case["right"])
    left_fingerprint = derived_fingerprints(left)
    right_fingerprint = derived_fingerprints(right)
    overlap = _hard_overlap(token_shingles(left), token_shingles(right))
    return {
        "id": str(case["id"]),
        "label": str(case["label"]),
        "exact_match": hashlib.sha256(left.encode("utf-8")).digest()
        == hashlib.sha256(right.encode("utf-8")).digest(),
        "simhash_distance": simhash_hamming_distance(
            str(left_fingerprint["token_simhash64"]),
            str(right_fingerprint["token_simhash64"]),
        ),
        "jaccard": round(float(overlap["jaccard"]), 6),
        "containment": round(float(overlap["containment"]), 6),
    }


def _classify(
    row: Dict[str, Any],
    *,
    simhash_threshold: int,
    jaccard_threshold: float,
    containment_threshold: float,
) -> bool:
    if bool(row["exact_match"]):
        return True
    return bool(
        int(row["simhash_distance"]) <= simhash_threshold
        and (
            float(row["jaccard"]) >= jaccard_threshold
            or float(row["containment"]) >= containment_threshold
        )
    )


def _sweep(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for simhash_threshold in SIMHASH_THRESHOLDS:
        for jaccard_threshold in JACCARD_THRESHOLDS:
            for containment_threshold in CONTAINMENT_THRESHOLDS:
                tp = fp = tn = fn = 0
                for row in rows:
                    expected = row["label"] == "hard_duplicate"
                    predicted = _classify(
                        row,
                        simhash_threshold=simhash_threshold,
                        jaccard_threshold=jaccard_threshold,
                        containment_threshold=containment_threshold,
                    )
                    if expected and predicted:
                        tp += 1
                    elif expected:
                        fn += 1
                    elif predicted:
                        fp += 1
                    else:
                        tn += 1
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                f1 = 2 * precision * recall / max(1e-12, precision + recall)
                results.append(
                    {
                        "simhash_threshold": simhash_threshold,
                        "jaccard_threshold": jaccard_threshold,
                        "containment_threshold": containment_threshold,
                        "true_positive": tp,
                        "false_positive": fp,
                        "true_negative": tn,
                        "false_negative": fn,
                        "precision": round(precision, 6),
                        "recall": round(recall, 6),
                        "f1": round(f1, 6),
                    }
                )
    return sorted(
        results,
        key=lambda row: (
            -row["f1"],
            -row["precision"],
            -row["recall"],
            row["simhash_threshold"],
            -row["jaccard_threshold"],
            -row["containment_threshold"],
        ),
    )


def _stage_b_row(uid: str, text: str) -> Dict[str, Any]:
    return {
        "chunk_uid": uid,
        "split": "train",
        "stage_a_pass": True,
        "bundle_id": "redundancy-benchmark",
        "repository_identity": "fixture/redundancy",
        "path": f"tests/generated/{uid}.py",
        "change_type": "modified",
        "content_type": "test",
        "chunk_kind": "function",
        "text": text,
    }


def _saturation_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    template = payload["saturation_template"]
    records = [str(value) for value in template["records"]]
    rows = []
    for size in template["sizes"]:
        scored = score_stage_b(
            [_stage_b_row(f"template-{index + 1:03d}", text) for index, text in enumerate(records[: int(size)])],
            quality_weight=0.8,
            redundancy_weight=0.2,
        )
        risks = [float(row["stage_b_evidence"]["soft_redundancy_risk"]) for row in scored]
        counts = [int(row["stage_b_evidence"]["soft_structural_match_count"]) for row in scored]
        rows.append(
            {
                "group_size": int(size),
                "mean_soft_redundancy_risk": round(sum(risks) / max(1, len(risks)), 6),
                "max_soft_redundancy_risk": round(max(risks, default=0.0), 6),
                "mean_structural_match_count": round(sum(counts) / max(1, len(counts)), 6),
            }
        )
    repeated_groups = [row for row in rows if row["group_size"] >= 2]
    risk_increases = all(
        right["mean_soft_redundancy_risk"] > left["mean_soft_redundancy_risk"]
        for left, right in zip(repeated_groups, repeated_groups[1:])
    )
    count_increases = all(
        right["mean_structural_match_count"] > left["mean_structural_match_count"]
        for left, right in zip(repeated_groups, repeated_groups[1:])
    )
    return {
        "groups": rows,
        "risk_strictly_increases_after_first_duplicate": risk_increases,
        "match_count_strictly_increases": count_increases,
        "interpretation": (
            "Current Stage-B risk is saturation-sensitive."
            if risk_increases
            else "Current Stage-B risk records match count but collapses one-or-more structural matches to the same 0.85 risk."
        ),
    }


def build(fixtures_path: Path, output_path: Path, md_output_path: Path) -> Dict[str, Any]:
    payload = load_json(fixtures_path)
    cases = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("Redundancy benchmark fixtures must contain a pairs list.")
    rows = [_pair_evidence(case) for case in cases]
    sweep = _sweep(rows)
    current = next(
        row
        for row in sweep
        if row["simhash_threshold"] == 3
        and row["jaccard_threshold"] == 0.75
        and row["containment_threshold"] == 0.88
    )
    for row in rows:
        row["current_hard_duplicate_prediction"] = _classify(
            row,
            simhash_threshold=3,
            jaccard_threshold=0.75,
            containment_threshold=0.88,
        )
        row["current_correct"] = bool(row["current_hard_duplicate_prediction"]) == (
            row["label"] == "hard_duplicate"
        )
    saturation = _saturation_evidence(payload)
    gaps = []
    if current["precision"] < 1.0:
        gaps.append("current_threshold_false_positive_on_labeled_fixture")
    if current["recall"] < 1.0:
        gaps.append("current_threshold_false_negative_on_labeled_fixture")
    if not saturation["risk_strictly_increases_after_first_duplicate"]:
        gaps.append("stage_b_soft_risk_not_saturation_magnitude_sensitive")
    report = {
        "schema_version": "redundancy-validity-benchmark-report-v1",
        "status": (
            "redundancy_benchmark_completed_with_known_gaps"
            if gaps
            else "redundancy_benchmark_passed_current_fixture"
        ),
        "claim_boundary": payload.get("claim_boundary"),
        "fixtures_path": str(fixtures_path),
        "summary": {
            "pair_count": len(rows),
            "hard_duplicate_count": sum(row["label"] == "hard_duplicate" for row in rows),
            "related_useful_count": sum(row["label"] == "related_useful" for row in rows),
            "independent_count": sum(row["label"] == "independent" for row in rows),
            "current_threshold": current,
            "best_fixture_threshold": sweep[0],
            "current_correct_count": sum(bool(row["current_correct"]) for row in rows),
        },
        "pairs": rows,
        "threshold_sweep_top10": sweep[:10],
        "saturation": saturation,
        "known_gaps": gaps,
        "required_next_evidence": [
            "expand_labeled_pairs_with_repository_disjoint_real_code",
            "measure_precision_recall_by_content_type_and_chunk_length",
            "validate_representative_dropout_on_real_duplicate_clusters",
            "replace_binary_structural_risk_with_calibrated_saturation_response",
            "keep_semantic_clone_detection_out_of_hard_gate_until_high_precision",
        ],
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    current = report["summary"]["current_threshold"]
    best = report["summary"]["best_fixture_threshold"]
    lines = [
        "# Redundancy Validity Benchmark",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Current Threshold",
        "",
        f"- Precision: `{current['precision']}`",
        f"- Recall: `{current['recall']}`",
        f"- F1: `{current['f1']}`",
        f"- False positives: `{current['false_positive']}`",
        f"- False negatives: `{current['false_negative']}`",
        "",
        "## Best Fixture Threshold",
        "",
        f"- SimHash: `{best['simhash_threshold']}`",
        f"- Jaccard: `{best['jaccard_threshold']}`",
        f"- Containment: `{best['containment_threshold']}`",
        f"- Precision / Recall / F1: `{best['precision']} / {best['recall']} / {best['f1']}`",
        "",
        "## Pair Results",
        "",
        "| Pair | Label | Exact | SimHash | Jaccard | Containment | Current Correct |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["pairs"]:
        lines.append(
            f"| `{row['id']}` | `{row['label']}` | `{row['exact_match']}` | "
            f"`{row['simhash_distance']}` | `{row['jaccard']}` | `{row['containment']}` | "
            f"`{row['current_correct']}` |"
        )
    lines.extend(["", "## Saturation", ""])
    for row in report["saturation"]["groups"]:
        lines.append(
            f"- Size `{row['group_size']}`: risk `{row['mean_soft_redundancy_risk']}`, "
            f"mean structural matches `{row['mean_structural_match_count']}`"
        )
    lines.extend(["", report["saturation"]["interpretation"], "", "## Known Gaps", ""])
    lines.extend([f"- `{gap}`" for gap in report["known_gaps"]] or ["- None on this bounded fixture."])
    lines.extend(["", "## Required Next Evidence", ""])
    lines.extend([f"- `{gap}`" for gap in report["required_next_evidence"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Redundancy validity benchmark.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.fixtures, args.output, args.md_output)
    print(
        {
            "status": report["status"],
            "summary": report["summary"],
            "known_gaps": report["known_gaps"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
