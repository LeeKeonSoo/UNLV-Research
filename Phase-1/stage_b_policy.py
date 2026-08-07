from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from stage_c_selection import STRUCTURAL_POLICY_REASON_CODES, select_chunks


JsonMap = dict[str, Any]
STAGE_B_STRUCTURAL_POLICY_REASON_CODES = STRUCTURAL_POLICY_REASON_CODES


def _stage_b_policy_id(value: object) -> object:
    if isinstance(value, str) and value.startswith("stage_c_"):
        return f"stage_b_{value.removeprefix('stage_c_')}"
    return value


def propose_stage_b_removals(
    chunks: Iterable[JsonMap], config: JsonMap
) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    selected, removed, legacy_audit = select_chunks(chunks, config)
    for row in (*selected, *removed):
        trace = row.pop("stage_c_selection", None)
        if isinstance(trace, dict):
            if "quality_policy_id" in trace:
                trace["quality_policy_id"] = _stage_b_policy_id(trace["quality_policy_id"])
            row["stage_b_policy"] = trace
        metadata = row.pop("stage_c_policy_metadata", None)
        if isinstance(metadata, dict):
            row["stage_b_policy_metadata"] = metadata
        visible = row.pop("stage_c_selector_visible", None)
        if isinstance(visible, dict):
            row["stage_b_selector_visible"] = visible
        quality = row.get("quality_retention_decision")
        if isinstance(quality, dict):
            if "policy_id" in quality:
                quality["policy_id"] = _stage_b_policy_id(quality["policy_id"])
            evaluated = quality.get("evaluated_policy_ids")
            if isinstance(evaluated, list):
                quality["evaluated_policy_ids"] = [
                    _stage_b_policy_id(policy_id) for policy_id in evaluated
                ]
    audit = {
        **legacy_audit,
        "owner_stage": "Stage B",
        "core_ids": ["redundancy", "quality"],
        "proposal_is_final_membership": False,
        "legacy_implementation_adapter": "stage_c_selection.select_chunks",
    }
    return selected, removed, audit
