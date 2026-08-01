#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import _coverage_impact_audit


def _composition_audit() -> dict[str, object]:
    return {"delta_from_raw": {"stage_c_curated": {"content_domain": {"token_share": {}}}}}


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
            "stage_c_selection": {
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
        minimum_chunk_chars=40,
        composition_audit=_composition_audit(),
    )
    assert audit["authority"] == "audit_only"
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
        minimum_chunk_chars=40,
        composition_audit=_composition_audit(),
    )
    assert failed["representative_linkage"]["passed"] is False
    assert failed["representative_linkage"]["missing_representative_chunk_uids"] == ["exact-copy"]

    license_only = [
        {
            "chunk_uid": "license-representative",
            "stage_a_record_id": "license-record",
            "stage_c_selection": {"removed_reason": "license_comment_only_chunk"},
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
        minimum_chunk_chars=40,
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
    print("[coverage-invariants] audit-only representative linkage: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
