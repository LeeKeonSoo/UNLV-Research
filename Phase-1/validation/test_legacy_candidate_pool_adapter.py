#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from adapt_legacy_candidate_pool import adapt_rows


def test_adapt_rows_preserves_legacy_source_identity_without_rights_promotion() -> None:
    # Given: a legacy collector row and source facts with unresolved rights.
    rows = [{"record_uid": "legacy-1", "text": "A sufficiently long legacy candidate.", "source_dataset_id": "fixture/legacy", "source_split": "train", "source_row_index": 7, "domain": "math", "pool_role": "raw_like"}]
    source = {"source_name": "fixture/legacy", "source_uri": "https://example.invalid/fixture", "collected_at": "2026-07-22T14:50:00+09:00", "rights": {"status": "unknown", "license": None}, "pii_context": "general", "language": {"code": "en", "confidence": None}, "partition": {"source_tier": "raw_like", "content_type": "general_text"}}

    # When: the legacy rows enter the current candidate boundary.
    adapted = adapt_rows(rows, source)

    # Then: source identity is retained and unknown rights remain unknown.
    assert adapted[0]["record_id"] == "legacy-1"
    assert adapted[0]["provenance"]["source_name"] == "fixture/legacy"
    assert adapted[0]["rights"] == {"status": "unknown", "license": None}
    assert adapted[0]["partition"]["source_split"] == "train"
    assert adapted[0]["partition"]["source_row_index"] == 7
    assert adapted[0]["partition"]["content_type"] == "general_text"
    assert adapted[0].get("artifact_context") is None


if __name__ == "__main__":
    test_adapt_rows_preserves_legacy_source_identity_without_rights_promotion()
    print("[legacy-candidate-adapter] source preservation without rights promotion: pass")
