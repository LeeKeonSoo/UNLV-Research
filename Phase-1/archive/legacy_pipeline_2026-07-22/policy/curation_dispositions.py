"""Canonical curation and training-budget dispositions."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


CURATION_RETAINED = "retained"
CURATION_REJECTED = "rejected"
CURATION_QUARANTINED = "quarantined"

BUDGET_NOT_REQUESTED = "not_requested"
BUDGET_SELECTED = "selected_for_training_budget"
BUDGET_NOT_SELECTED = "budget_not_selected"


def annotate_retained_pool(
    records: Iterable[Dict[str, Any]],
    *,
    selected_ids: Set[str],
    budget_applied: bool,
) -> List[Dict[str, Any]]:
    """Annotate Stage-A-pass records without treating budget exclusion as rejection."""
    annotated: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        uid = str(row["chunk_uid"])
        selected = uid in selected_ids
        row["curation_decision"] = {
            "curation_disposition": CURATION_RETAINED,
            "curation_reason": "passed_stage_a_hard_gate",
            "training_budget_disposition": (
                BUDGET_SELECTED
                if budget_applied and selected
                else BUDGET_NOT_SELECTED
                if budget_applied
                else BUDGET_NOT_REQUESTED
            ),
            "budget_exclusion_is_rejection": False,
        }
        annotated.append(row)
    return annotated


def disposition_summary(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    curation_counts: Dict[str, int] = {}
    budget_counts: Dict[str, int] = {}
    for row in rows:
        decision = row.get("curation_decision") or {}
        curation = str(decision.get("curation_disposition") or "unknown")
        budget = str(decision.get("training_budget_disposition") or "unknown")
        curation_counts[curation] = curation_counts.get(curation, 0) + 1
        budget_counts[budget] = budget_counts.get(budget, 0) + 1
    return {
        "record_count": len(rows),
        "curation_disposition_counts": dict(sorted(curation_counts.items())),
        "training_budget_disposition_counts": dict(sorted(budget_counts.items())),
        "budget_not_selected_is_rejection": False,
    }
