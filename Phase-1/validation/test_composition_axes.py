#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from composition_audit import annotate_record, annotate_records, build_composition_audit


def test_four_axes_describe_text_without_granting_selection_authority() -> None:
    code = annotate_record({"text": "def add(left, right):\n    return left + right\nimport math"})
    dialogue = annotate_record({"text": "Ava: Are you coming?\nBen: Yes, after class.\nAva: Great, see you then."})
    navigation = annotate_record({"text": "Cookie Preferences\nAccept All\nReject All\nManage Preferences"})

    assert code["content_domain"] == "code"
    assert code["document_format"] == "code"
    assert dialogue["document_format"] == "dialogue"
    assert dialogue["document_function"] == "discussion"
    assert navigation["document_function"] == "navigation_ui"

    audit = build_composition_audit(
        {
            "raw_input": annotate_records([{"text": "Ava: Are you coming?\nBen: Yes, after class.\nAva: Great, see you then.", "token_proxy": 10}]),
            "stage_b_pass": annotate_records([{"text": "Ava: Are you coming?\nBen: Yes, after class.\nAva: Great, see you then.", "token_proxy": 10}]),
            "stage_c_curated": annotate_records([{"text": "Ava: Are you coming?\nBen: Yes, after class.\nAva: Great, see you then.", "token_proxy": 10}]),
        }
    )
    assert audit["authority"] == "audit_only"
    assert audit["consumed_by_stage_c"] is False
    assert "document_format" in audit["stages"]["raw_input"]
    assert "stage_c_curated" not in audit["delta_from_raw"]
    assert audit["stage_units"] == {
        "raw_input": "record",
        "stage_b_pass": "chunk",
        "stage_c_curated": "chunk",
    }
    assert audit["excluded_cross_unit_deltas"]["stage_c_curated"] == "record_to_chunk_delta_not_emitted"
    assert "document_function" in audit["delta_from_stage_b_pass"]["stage_c_curated"]


if __name__ == "__main__":
    test_four_axes_describe_text_without_granting_selection_authority()
    print("[composition-axes] four-axis audit-only labeling: pass")
