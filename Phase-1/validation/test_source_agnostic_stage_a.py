#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.candidate_processing import process_candidate


def test_text_only_v2_releases_a_complete_text_without_source_or_rights_metadata() -> None:
    record = process_candidate(
        {
            "id": "text-only",
            "text": "A complete text-only training candidate contains enough contextual content to form a meaningful learning unit without any source metadata.",
        },
        stage_a_policy="text_only_v2",
    )

    assert record["release_eligibility"]["eligible"] is True
    assert record["quarantine"]["status"] == "release_candidate"
    assert record["provenance"]["source_name"] == "unknown"
    assert record["rights"]["status"] == "unknown"
    assert record["hazards"]["diagnostics"]["audit_only"] is True


if __name__ == "__main__":
    test_text_only_v2_releases_a_complete_text_without_source_or_rights_metadata()
    print("[source-agnostic-stage-a] text-only v2 releases source-free input: pass")
