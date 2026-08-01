#!/usr/bin/env python3
"""Evaluate frozen hard-near-duplicate threshold arms on silver holdout."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_STAGE_A = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_a_code_domain_v2_combined"
    / "train"
    / "stage_a_pass.jsonl"
)
DEFAULT_HOLDOUT = Path("configs") / "temporal_code_redundancy_silver_holdout_v1.json"
DEFAULT_ARMS = Path("configs") / "temporal_code_hard_near_duplicate_threshold_arms_v1.json"
DEFAULT_PAIRS = OUTPUT_DIR / "validation" / "redundancy_silver_holdout_pairs.jsonl"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_silver_holdout_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_silver_holdout_report.md"


def _load_calibration_module():
    path = Path(__file__).resolve().parents[2] / "174_build_real_corpus_redundancy_calibration.py"
    spec = importlib.util.spec_from_file_location("real_corpus_redundancy_calibration", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load calibration module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    stage_a_path: Path,
    holdout_path: Path,
    arms_path: Path,
    pairs_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    module = _load_calibration_module()
    holdout = load_json(holdout_path)
    arms = load_json(arms_path)
    all_rows = list(_jsonl(stage_a_path))
    by_uid = {str(row["chunk_uid"]): row for row in all_rows}
    sources = [by_uid[str(uid)] for uid in holdout["source_chunk_uids"]]
    holdout_repositories = {str(row.get("repository_identity") or "") for row in sources}
    eligible_rows = [
        row
        for row in all_rows
        if str(row.get("repository_identity") or "") in holdout_repositories
    ]
    pairs, source_metadata = module._build_pairs_from_sources(eligible_rows, sources)
    arm_results = {}
    for name, threshold in arms["arms"].items():
        metrics = module._metrics(pairs, threshold)
        near_only = [
            row
            for row in pairs
            if row["transformation"] != "exact_copy"
        ]
        arm_results[name] = {
            "threshold": {
                "simhash_threshold": int(threshold["simhash_threshold"]),
                "jaccard_threshold": float(threshold["jaccard_threshold"]),
                "containment_threshold": float(threshold["containment_threshold"]),
            },
            "metrics": metrics,
            "near_only_metrics": module._metrics(near_only, threshold),
            "stratified": module._stratified(pairs, threshold),
            "passes_dropout_gate": metrics["useful_data_dropout_rate"] <= 0.05,
            "passes_precision_gate": metrics["precision"] >= 0.98,
        }
        arm_results[name]["eligible_after_holdout"] = bool(
            arm_results[name]["passes_dropout_gate"]
            and arm_results[name]["passes_precision_gate"]
        )

    eligible = [
        (name, row)
        for name, row in arm_results.items()
        if row["eligible_after_holdout"]
    ]
    eligible.sort(
        key=lambda item: (
            -item[1]["near_only_metrics"]["recall"],
            item[1]["metrics"]["useful_data_dropout_rate"],
            -item[1]["metrics"]["precision"],
            item[0],
        )
    )
    candidate = eligible[0][0] if eligible else None
    current_name = str(arms["current_arm"])
    report = {
        "schema_version": "redundancy-silver-holdout-report-v1",
        "status": (
            "redundancy_holdout_candidate_identified_cluster_audit_required"
            if candidate
            else "redundancy_holdout_no_candidate_passed"
        ),
        "claim_boundary": (
            "Independent repository-disjoint silver holdout. Passing this report does not promote "
            "a Stage-A threshold without cluster-level representative-dropout evidence."
        ),
        "holdout_contract": str(holdout_path),
        "threshold_arms": str(arms_path),
        "source_metadata": source_metadata,
        "calibration_repository_overlap": int(holdout["calibration_repository_overlap"]),
        "pair_count": len(pairs),
        "arm_results": arm_results,
        "current_arm": current_name,
        "candidate_after_holdout": candidate,
        "candidate_improvement_vs_current": (
            {
                "near_only_recall_delta": round(
                    arm_results[candidate]["near_only_metrics"]["recall"]
                    - arm_results[current_name]["near_only_metrics"]["recall"],
                    6,
                ),
                "dropout_delta": round(
                    arm_results[candidate]["metrics"]["useful_data_dropout_rate"]
                    - arm_results[current_name]["metrics"]["useful_data_dropout_rate"],
                    6,
                ),
            }
            if candidate
            else None
        ),
        "promotion_blockers": [
            "cluster_level_representative_dropout_not_measured",
            "silver_labels_are_not_human_validated_semantic_clone_ground_truth",
            "target_model_ablation_not_run",
        ],
        "next_actions": [
            "build cluster-level dropout audit for current and holdout candidate",
            "freeze a Stage-A development ablation without changing canonical outputs",
            "measure Stage-B pool and Coverage changes caused by the candidate",
            "only then decide whether candidate may enter proxy-model training",
        ],
    }
    _write_jsonl(pairs_path, pairs)
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Redundancy Silver Holdout",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        f"- Sources: `{report['source_metadata']['source_count']}`",
        f"- Repositories: `{report['source_metadata']['source_repository_count']}`",
        f"- Pairs: `{report['pair_count']}`",
        f"- Calibration repository overlap: `{report['calibration_repository_overlap']}`",
        "",
        "## Arms",
        "",
        "| Arm | Precision | Recall | Near Recall | Dropout | Eligible |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in report["arm_results"].items():
        lines.append(
            f"| `{name}` | `{row['metrics']['precision']}` | `{row['metrics']['recall']}` | "
            f"`{row['near_only_metrics']['recall']}` | `{row['metrics']['useful_data_dropout_rate']}` | "
            f"`{row['eligible_after_holdout']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Candidate after holdout: `{report['candidate_after_holdout']}`",
            f"- Improvement versus current: `{report['candidate_improvement_vs_current']}`",
            "",
            "## Promotion Blockers",
            "",
        ]
    )
    lines.extend([f"- `{value}`" for value in report["promotion_blockers"]])
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {value}" for value in report["next_actions"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Redundancy silver holdout.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.stage_a,
        args.holdout,
        args.arms,
        args.pairs,
        args.output,
        args.md_output,
    )
    print(
        {
            "status": report["status"],
            "candidate": report["candidate_after_holdout"],
            "arm_results": {
                name: {
                    "precision": row["metrics"]["precision"],
                    "near_recall": row["near_only_metrics"]["recall"],
                    "dropout": row["metrics"]["useful_data_dropout_rate"],
                    "eligible": row["eligible_after_holdout"],
                }
                for name, row in report["arm_results"].items()
            },
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
