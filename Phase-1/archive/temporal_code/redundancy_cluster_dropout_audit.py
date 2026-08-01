#!/usr/bin/env python3
"""Audit cluster-level data loss for hard-near-duplicate threshold arms."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_chunks import _hard_overlap, token_shingles
from ingestion.code_fingerprints import simhash_hamming_distance
from ingestion.code_selection import token_proxy_count


DEFAULT_STAGE0 = OUTPUT_DIR / "temporal_code_collection" / "stage0_code_domain_v2_combined"
DEFAULT_ARMS = Path("configs") / "temporal_code_hard_near_duplicate_threshold_arms_v1.json"
DEFAULT_SCORED = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_b_code_domain_v2"
    / "train_scored_full_selector.jsonl"
)
DEFAULT_SELECTED = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_b_code_domain_v2"
    / "curated_v2_equal_budget.jsonl"
)
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_cluster_dropout_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_cluster_dropout_audit.md"
AUDIT_ARMS = ("current", "zero_dropout_candidate")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _prepare(stage0_dir: Path) -> List[Dict[str, Any]]:
    module = importlib.import_module("74_run_temporal_code_stage_a_smoke")
    decisions = module.chunks_for_split(stage0_dir, "train")["decisions"]
    rows = []
    for row in decisions:
        if not bool(row.get("duplicate_representative_eligible")):
            continue
        rows.append(
            {
                **row,
                "shingles": token_shingles(str(row.get("text") or "")),
                "token_simhash64": str((row.get("stage_a_evidence") or {}).get("token_simhash64") or ""),
                "text_sha256": str((row.get("stage_a_evidence") or {}).get("text_sha256") or ""),
            }
        )
    return sorted(rows, key=lambda row: str(row["chunk_uid"]))


def _simulate(rows: List[Dict[str, Any]], threshold: Dict[str, Any]) -> Dict[str, Any]:
    accepted = []
    exact_representatives: Dict[str, Dict[str, Any]] = {}
    rejected = []
    for row in rows:
        exact = row["text_sha256"]
        if exact in exact_representatives:
            rejected.append(
                {
                    "chunk_uid": row["chunk_uid"],
                    "reason": "exact_duplicate",
                    "representative_uid": exact_representatives[exact]["chunk_uid"],
                    "simhash_distance": 0,
                    "jaccard": 1.0,
                    "containment": 1.0,
                }
            )
            continue
        match = None
        for representative in accepted:
            if not row["token_simhash64"] or not representative["token_simhash64"]:
                continue
            distance = simhash_hamming_distance(row["token_simhash64"], representative["token_simhash64"])
            if distance > int(threshold["simhash_threshold"]):
                continue
            overlap = _hard_overlap(row["shingles"], representative["shingles"])
            if (
                overlap["jaccard"] >= float(threshold["jaccard_threshold"])
                or overlap["containment"] >= float(threshold["containment_threshold"])
            ):
                match = {
                    "chunk_uid": row["chunk_uid"],
                    "reason": "hard_near_duplicate",
                    "representative_uid": representative["chunk_uid"],
                    "simhash_distance": distance,
                    "jaccard": round(float(overlap["jaccard"]), 6),
                    "containment": round(float(overlap["containment"]), 6),
                }
                break
        if match is not None:
            rejected.append(match)
            continue
        accepted.append(row)
        exact_representatives[exact] = row
    return {
        "accepted_ids": {str(row["chunk_uid"]) for row in accepted},
        "rejected": rejected,
        "rejected_by_uid": {str(row["chunk_uid"]): row for row in rejected},
    }


def _loss_summary(
    lost_ids: set[str],
    candidate: Dict[str, Any],
    rows_by_uid: Dict[str, Dict[str, Any]],
    scored_by_uid: Dict[str, Dict[str, Any]],
    selected_ids: set[str],
) -> Dict[str, Any]:
    lost_rows = [rows_by_uid[uid] for uid in sorted(lost_ids)]
    selected_lost = [uid for uid in lost_ids if uid in selected_ids]
    token_count = sum(token_proxy_count(str(row.get("text") or "")) for row in lost_rows)
    selected_tokens = sum(
        int((scored_by_uid.get(uid, {}).get("stage_b_evidence") or {}).get("token_proxy_count") or 0)
        for uid in selected_lost
    )
    quality_values = [
        float((scored_by_uid.get(uid, {}).get("stage_b_evidence") or {}).get("code_quality_proxy"))
        for uid in lost_ids
        if (scored_by_uid.get(uid, {}).get("stage_b_evidence") or {}).get("code_quality_proxy") is not None
    ]
    examples = []
    for uid in sorted(lost_ids):
        match = candidate["rejected_by_uid"].get(uid) or {}
        row = rows_by_uid[uid]
        examples.append(
            {
                "chunk_uid": uid,
                "representative_uid": match.get("representative_uid"),
                "reason": match.get("reason"),
                "simhash_distance": match.get("simhash_distance"),
                "jaccard": match.get("jaccard"),
                "containment": match.get("containment"),
                "repository_identity": row.get("repository_identity"),
                "path": row.get("path"),
                "content_type": row.get("content_type"),
                "chunk_kind": row.get("chunk_kind"),
                "was_stage_b_selected": uid in selected_ids,
                "stage_b_quality_proxy": (
                    scored_by_uid.get(uid, {}).get("stage_b_evidence") or {}
                ).get("code_quality_proxy"),
            }
        )
    return {
        "lost_record_count": len(lost_ids),
        "lost_token_proxy_count": token_count,
        "lost_stage_b_selected_count": len(selected_lost),
        "lost_stage_b_selected_token_proxy_count": selected_tokens,
        "lost_content_type_counts": dict(
            sorted(Counter(str(row.get("content_type") or "unknown") for row in lost_rows).items())
        ),
        "lost_repository_count": len(
            {str(row.get("repository_identity") or "") for row in lost_rows}
        ),
        "mean_lost_stage_b_quality_proxy": (
            round(sum(quality_values) / len(quality_values), 6) if quality_values else None
        ),
        "examples": examples[:50],
    }


def build(
    stage0_dir: Path,
    arms_path: Path,
    scored_path: Path,
    selected_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    arms = load_json(arms_path)
    rows = _prepare(stage0_dir)
    rows_by_uid = {str(row["chunk_uid"]): row for row in rows}
    scored_by_uid = {str(row["chunk_uid"]): row for row in _jsonl(scored_path)}
    selected_ids = {str(row["chunk_uid"]) for row in _jsonl(selected_path)}
    simulations = {
        name: _simulate(rows, arms["arms"][name])
        for name in AUDIT_ARMS
    }
    current = simulations["current"]
    challenger = simulations["zero_dropout_candidate"]
    lost = current["accepted_ids"] - challenger["accepted_ids"]
    gained = challenger["accepted_ids"] - current["accepted_ids"]
    loss = _loss_summary(lost, challenger, rows_by_uid, scored_by_uid, selected_ids)
    candidate_threshold = arms["arms"]["zero_dropout_candidate"]
    report = {
        "schema_version": "redundancy-cluster-dropout-audit-v1",
        "status": "redundancy_cluster_dropout_audit_ready",
        "claim_boundary": (
            "Outcome-free corpus audit comparing frozen Stage-A threshold arms. "
            "Stage-B selection evidence is used only to quantify threatened retained data, not Utility."
        ),
        "source_stage0": str(stage0_dir),
        "threshold_arms": str(arms_path),
        "locally_eligible_chunk_count": len(rows),
        "arms": {
            name: {
                "threshold": arms["arms"][name],
                "accepted_count": len(simulation["accepted_ids"]),
                "rejected_count": len(simulation["rejected"]),
                "exact_rejected_count": sum(
                    row["reason"] == "exact_duplicate" for row in simulation["rejected"]
                ),
                "near_rejected_count": sum(
                    row["reason"] == "hard_near_duplicate" for row in simulation["rejected"]
                ),
            }
            for name, simulation in simulations.items()
        },
        "challenger": "zero_dropout_candidate",
        "challenger_threshold": candidate_threshold,
        "current_accepted_lost_by_challenger": loss,
        "challenger_accepted_not_in_current": {
            "record_count": len(gained),
            "chunk_uids": sorted(gained)[:50],
        },
        "decision": (
            "hold_challenger"
            if loss["lost_stage_b_selected_count"] > 0
            else "eligible_for_stage_a_development_ablation"
        ),
        "decision_reasons": [
            "silver_holdout_precision_and_dropout_gate_failed",
            (
                "challenger_removes_previously_selected_stage_b_records"
                if loss["lost_stage_b_selected_count"] > 0
                else "no_previously_selected_stage_b_records_removed"
            ),
        ],
        "next_actions": [
            "inspect semantic-change false-positive families and add structural guards",
            "do not relax the canonical Stage-A threshold",
            "move template and containment saturation recovery to Stage-B soft evidence",
            "implement count-sensitive saturation arms and rerun Stage-B ablations",
        ],
        "utility_scope": "Stage C only; not consumed by this audit",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    loss = report["current_accepted_lost_by_challenger"]
    lines = [
        "# Redundancy Cluster Dropout Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        f"- Locally eligible chunks: `{report['locally_eligible_chunk_count']}`",
        "",
        "## Arms",
        "",
        "| Arm | Accepted | Rejected | Exact | Near |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, row in report["arms"].items():
        lines.append(
            f"| `{name}` | `{row['accepted_count']}` | `{row['rejected_count']}` | "
            f"`{row['exact_rejected_count']}` | `{row['near_rejected_count']}` |"
        )
    lines.extend(
        [
            "",
            "## Challenger Loss",
            "",
            f"- Lost current records: `{loss['lost_record_count']}`",
            f"- Lost token proxy: `{loss['lost_token_proxy_count']}`",
            f"- Lost Stage-B-selected records: `{loss['lost_stage_b_selected_count']}`",
            f"- Lost Stage-B-selected token proxy: `{loss['lost_stage_b_selected_token_proxy_count']}`",
            f"- Content types: `{loss['lost_content_type_counts']}`",
            f"- Mean Selection Value Evidence: `{loss['mean_lost_stage_b_quality_proxy']}`",
            "",
            "## Decision",
            "",
            f"`{report['decision']}`",
            "",
        ]
    )
    lines.extend([f"- `{reason}`" for reason in report["decision_reasons"]])
    lines.extend(["", "## Next Actions", ""])
    lines.extend([f"- {value}" for value in report["next_actions"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Redundancy cluster dropout audit.")
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0)
    parser.add_argument("--arms", type=Path, default=DEFAULT_ARMS)
    parser.add_argument("--scored", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.stage0_dir,
        args.arms,
        args.scored,
        args.selected,
        args.output,
        args.md_output,
    )
    print(
        {
            "status": report["status"],
            "decision": report["decision"],
            "loss": report["current_accepted_lost_by_challenger"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
