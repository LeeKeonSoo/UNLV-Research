#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import _coverage_impact_audit


def _composition_audit() -> dict[str, object]:
    return {
        "delta_from_stage_b_pass": {
            "stage_c_curated": {"content_domain": {"token_share": {}}}
        },
    }


def main() -> int:
    selected = [{"chunk_uid": "representative", "stage_a_record_id": "representative-record"}]
    rejected = [
        {
            "chunk_uid": "exact-copy",
            "stage_b_hard_gate_reasons": ["normalized_exact_duplicate"],
            "stage_b_decision": {"representative_chunk_uid": "representative"},
        }
    ]
    not_selected = [
        {
            "chunk_uid": "scaffold-copy",
            "stage_a_record_id": "scaffold-record",
            "stage_b_policy": {
                "removed_reason": "structural_scaffold_representative_retained",
                "representative_chunk_uid": "representative",
            },
        }
    ]
    audit = _coverage_impact_audit(
        passed=selected,
        selected=selected,
        rejected=rejected,
        not_selected=not_selected,
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit(),
    )
    assert audit["authority"] == "materialization_invariant"
    assert audit["selector_consumes_this_audit"] is False
    assert audit["metadata_strata_or_target_mix_used"] is False
    assert audit["representative_linkage"]["required_removed_chunks"] == 2
    assert audit["representative_linkage"]["passed"] is True
    assert audit["zero_survivor_invariant"]["passed"] is True
    assert audit["rule_interaction_audit"]["passed"] is True
    assert audit["passed"] is True

    rejected[0]["stage_b_decision"] = {}
    failed = _coverage_impact_audit(
        passed=selected,
        selected=selected,
        rejected=rejected,
        not_selected=not_selected,
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit(),
    )
    assert failed["representative_linkage"]["passed"] is False
    assert failed["representative_linkage"]["missing_representative_chunk_uids"] == ["exact-copy"]

    license_only = [
        {
            "chunk_uid": "license-representative",
            "stage_a_record_id": "license-record",
            "stage_b_policy": {"removed_reason": "license_comment_only_chunk"},
        }
    ]
    resolved = _coverage_impact_audit(
        passed=selected,
        selected=selected,
        rejected=[
            {
                "chunk_uid": "license-copy",
                "stage_b_hard_gate_reasons": ["normalized_exact_duplicate"],
                "stage_b_decision": {"representative_chunk_uid": "license-representative"},
            }
        ],
        not_selected=license_only,
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit(),
    )
    assert resolved["passed"] is True
    assert resolved["representative_linkage"]["resolved_by_non_payload_removal"] == [
        {
            "chunk_uid": "license-copy",
            "representative_chunk_uid": "license-representative",
            "reason_code": "normalized_exact_duplicate",
            "representative_removed_reason": "license_comment_only_chunk",
        }
    ]

    chained = _coverage_impact_audit(
        passed=[
            {"chunk_uid": "final-representative", "stage_a_record_id": "final-record"},
            {"chunk_uid": "intermediate-representative", "stage_a_record_id": "intermediate-record"},
        ],
        selected=[{"chunk_uid": "final-representative", "stage_a_record_id": "final-record"}],
        rejected=[
            {
                "chunk_uid": "exact-copy-with-intermediate-link",
                "stage_b_hard_gate_reasons": ["normalized_exact_duplicate"],
                "stage_b_decision": {"representative_chunk_uid": "intermediate-representative"},
            }
        ],
        not_selected=[
            {
                "chunk_uid": "intermediate-representative",
                "stage_a_record_id": "intermediate-record",
                "stage_b_policy": {
                    "removed_reason": "near_duplicate_representative_retained",
                    "representative_chunk_uid": "final-representative",
                },
            }
        ],
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit(),
    )
    assert chained["passed"] is True
    assert chained["representative_linkage"]["resolved_by_representative_chain"] == [
        {
            "chunk_uid": "exact-copy-with-intermediate-link",
            "reason_code": "normalized_exact_duplicate",
            "representative_chain": ["intermediate-representative", "final-representative"],
            "terminal_chunk_uid": "final-representative",
        }
    ]

    quality_removed = _coverage_impact_audit(
        passed=[{"chunk_uid": "quality-fail", "stage_a_record_id": "quality-record"}],
        selected=[],
        rejected=[],
        not_selected=[
            {
                "chunk_uid": "quality-fail",
                "stage_a_record_id": "quality-record",
                "stage_b_policy": {
                    "removed_reason": "quality_normal_qualified_fail",
                    "failed_policy_ids": ["q3_substantive_payload"],
                },
            }
        ],
        span_transformations=[],
        minimum_residual_chars=40,
        composition_audit=_composition_audit(),
    )
    assert quality_removed["zero_survivor_invariant"]["passed"] is True
    assert quality_removed["passed"] is True

    print("[coverage-invariants] materialization representative linkage: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
