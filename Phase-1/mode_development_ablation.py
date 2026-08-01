from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from composition_audit import annotate_records, build_composition_audit


JsonMap = dict[str, Any]


def _require_candidate_report(report: JsonMap, schema_version: str, label: str) -> None:
    if report.get("schema_version") != schema_version:
        raise RuntimeError(f"{label} requires {schema_version}")
    if report.get("runtime_authorization") != "none_candidate_cannot_select_or_remove":
        raise RuntimeError(f"{label} must not have runtime authorization")


def _group_by_chunk(group_memberships: JsonMap, weak_rows: list[JsonMap]) -> dict[str, str]:
    weak_uids = {str(row.get("chunk_uid") or "") for row in weak_rows}
    if "" in weak_uids or len(weak_uids) != len(weak_rows):
        raise RuntimeError("Mode ablation requires unique non-empty Weak chunk_uid values")
    membership: dict[str, str] = {}
    for group_id, raw_uids in group_memberships.items():
        if not isinstance(group_id, str) or not group_id or not isinstance(raw_uids, list):
            raise RuntimeError("Mode ablation requires string group IDs and chunk UID lists")
        for raw_uid in raw_uids:
            if not isinstance(raw_uid, str) or not raw_uid:
                raise RuntimeError("Mode ablation memberships require non-empty chunk UIDs")
            if raw_uid in membership:
                raise RuntimeError(f"Mode ablation chunk has multiple group memberships: {raw_uid}")
            membership[raw_uid] = group_id
    if set(membership) != weak_uids:
        raise RuntimeError("Mode ablation group membership must cover every Weak chunk exactly once")
    return membership


def _group_ids(rows: Any, label: str) -> set[str]:
    if not isinstance(rows, list):
        raise RuntimeError(f"Mode ablation {label} must be a list")
    group_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise RuntimeError(f"Mode ablation {label} requires non-empty group_id values")
        group_ids.add(row["group_id"])
    return group_ids


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _arm_summary(rows: list[JsonMap], removed: list[JsonMap], path: Path) -> JsonMap:
    _write_jsonl(path, rows)
    reasons = Counter(str(row["reason_code"]) for row in removed)
    return {
        "chunks": len(rows),
        "token_proxy": sum(int(row.get("token_proxy") or len(str(row.get("text") or "").split())) for row in rows),
        "dataset_path": str(path),
        "removed_reasons": dict(sorted(reasons.items())),
    }


def materialize_mode_development_arms(
    *,
    weak_rows: Iterable[JsonMap],
    group_memberships: JsonMap,
    mid_report: JsonMap,
    hard_plan: JsonMap,
    output_dir: Path,
) -> JsonMap:
    """Materialize frozen candidate-only Weak, Mid, and Hard development arms."""
    weak = [dict(row) for row in weak_rows]
    membership = _group_by_chunk(group_memberships, weak)
    _require_candidate_report(mid_report, "mid-quality-development-report-v1", "Mode ablation Mid report")
    _require_candidate_report(hard_plan, "hard-quality-candidate-plan-v1", "Mode ablation Hard plan")

    mid_groups = mid_report.get("groups")
    if not isinstance(mid_groups, list):
        raise RuntimeError("Mode ablation Mid report requires groups")
    mid_decisions = {
        str(row["group_id"]): str(row.get("decision") or "")
        for row in mid_groups
        if isinstance(row, dict) and isinstance(row.get("group_id"), str)
    }
    if set(mid_decisions) != set(group_memberships):
        raise RuntimeError("Mode ablation Mid report must decide every frozen group")
    mid_removed_groups = {group_id for group_id, decision in mid_decisions.items() if decision == "candidate_remove"}
    mid_rows = [row for row in weak if membership[str(row["chunk_uid"])] not in mid_removed_groups]
    mid_removed = [
        {"chunk_uid": row["chunk_uid"], "reason_code": "mid_quality_calibrated_non_positive_candidate"}
        for row in weak
        if membership[str(row["chunk_uid"])] in mid_removed_groups
    ]

    selected_group_ids = _group_ids(hard_plan.get("selected_groups"), "Hard selected_groups")
    excluded = hard_plan.get("excluded_groups")
    if not isinstance(excluded, list):
        raise RuntimeError("Mode ablation Hard plan requires excluded_groups")
    excluded_reasons = {
        str(row["group_id"]): str(row.get("reason_code") or "")
        for row in excluded
        if isinstance(row, dict) and isinstance(row.get("group_id"), str)
    }
    if selected_group_ids & set(excluded_reasons) or selected_group_ids | set(excluded_reasons) != set(group_memberships):
        raise RuntimeError("Mode ablation Hard plan must partition every frozen group")
    if selected_group_ids & mid_removed_groups:
        raise RuntimeError("Mode ablation Hard plan must remain a subset of Mid survivors")
    hard_rows = [row for row in weak if membership[str(row["chunk_uid"])] in selected_group_ids]
    hard_removed = [
        {"chunk_uid": row["chunk_uid"], "reason_code": excluded_reasons[membership[str(row["chunk_uid"])]]}
        for row in weak
        if membership[str(row["chunk_uid"])] not in selected_group_ids
    ]

    arms = {
        "weak": _arm_summary(weak, [], output_dir / "weak.jsonl"),
        "mid": _arm_summary(mid_rows, mid_removed, output_dir / "mid.jsonl"),
        "hard": _arm_summary(hard_rows, hard_removed, output_dir / "hard.jsonl"),
    }
    composition = build_composition_audit(
        {
            "raw_input": annotate_records([dict(row) for row in weak]),
            "weak": annotate_records([dict(row) for row in weak]),
            "mid": annotate_records([dict(row) for row in mid_rows]),
            "hard": annotate_records([dict(row) for row in hard_rows]),
        }
    )
    report = {
        "schema_version": "mode-development-ablation-report-v1",
        "status": "candidate_only_development_artifact",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "arms": arms,
        "composition_audit": composition,
        "claim_boundary": "These are frozen development arms. They are separate future training inputs, not active runtime materializations.",
    }
    report_path = output_dir / "mode_development_ablation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
