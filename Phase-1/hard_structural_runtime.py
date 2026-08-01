from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from inline_license_comment_block_compaction import build_plan as build_license_block_plan
from inline_license_comment_block_compaction import materialize_candidate_plan as compact_license_blocks
from inline_license_header_compaction import build_plan as build_license_header_plan
from inline_license_header_compaction import materialize_candidate_plan as compact_license_headers
from span_level_template_compaction import build_plan as build_repeated_span_plan
from span_level_template_compaction import materialize_candidate_plan as compact_repeated_spans


JsonMap = dict[str, Any]
HARD_POLICY_IDS = (
    "stage_c_inline_license_header_candidate",
    "stage_c_inline_license_comment_block_candidate",
    "stage_c_repeated_span_template_candidate",
)


def _residual_is_valid(row: JsonMap, minimum_chunk_chars: int) -> bool:
    return len(str(row.get("text") or "").strip()) >= minimum_chunk_chars


def apply_development_hard_policies(
    rows: Iterable[JsonMap], *, minimum_chunk_chars: int
) -> JsonMap:
    """Apply the frozen Hard-v1 span candidates only in a development run.

    The function preserves every chunk and changes only declared non-payload spans.
    N4 must validate the resulting profile before a production curation run may use it.
    """
    if minimum_chunk_chars < 1:
        raise ValueError("Hard compaction requires a positive Stage-B minimum_chunk_chars")

    original_rows = [dict(row) for row in rows]
    header_plan = build_license_header_plan(
        original_rows, minimum_residual_chars=minimum_chunk_chars
    )
    header_result = compact_license_headers(original_rows, header_plan)
    block_plan = build_license_block_plan(
        header_result["records"], minimum_residual_chars=minimum_chunk_chars
    )
    block_result = compact_license_blocks(header_result["records"], block_plan)
    repeated_plan = build_repeated_span_plan(
        block_result["records"], minimum_residual_chars=minimum_chunk_chars
    )
    repeated_result = compact_repeated_spans(block_result["records"], repeated_plan)
    transformations = [
        *header_result["transformations"],
        *block_result["transformations"],
        *repeated_result["transformations"],
    ]
    transformed_chunk_uids = {str(item["chunk_uid"]) for item in transformations}
    invalid_residuals = [
        str(row.get("chunk_uid") or "unknown")
        for row in repeated_result["records"]
        if str(row.get("chunk_uid") or "unknown") in transformed_chunk_uids
        and not _residual_is_valid(row, minimum_chunk_chars)
    ]
    if invalid_residuals:
        raise RuntimeError(
            "Hard span compaction produced a residual below the declared Stage-B boundary: "
            + ", ".join(invalid_residuals)
        )
    return {
        "schema_version": "hard-structural-development-runtime-v1",
        "status": "development_only_pending_n4_ablation",
        "policy_ids": list(HARD_POLICY_IDS),
        "records": repeated_result["records"],
        "transformations": transformations,
        "residual_payload": {
            "minimum_chunk_chars": minimum_chunk_chars,
            "rewritten_chunks_checked": len(transformed_chunk_uids),
            "invalid_residual_chunk_uids": invalid_residuals,
            "passed": not invalid_residuals,
        },
        "runtime_authorization": "development_only_pending_n4_ablation",
    }
