#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_json_batch_candidate_pool import collect_rows


def test_collect_rows_preserves_batch_provenance_without_domain_policy() -> None:
    with TemporaryDirectory() as temporary:
        source = Path(temporary)
        (source / "batch_000.json").write_text(
            json.dumps([{"id": "general-1", "text": "A sufficiently long general text record for a domain-neutral curation replay.", "source_metadata": {"url": "https://example.invalid/original"}, "source_dataset": "fixture/general", "license": "ODC-By-1.0"}]),
            encoding="utf-8",
        )
        rows = collect_rows(source, collected_at="2026-07-27T00:00:00Z", limit=10)

    assert len(rows) == 1
    assert rows[0]["provenance"]["source_name"] == "fixture/general"
    assert rows[0]["provenance"]["source_uri"] == "https://example.invalid/original"
    assert rows[0]["rights"] == {"status": "allowed", "license": "ODC-By-1.0"}
    assert rows[0]["partition"]["content_type"] == "general_text"
    assert rows[0].get("artifact_context") is None


if __name__ == "__main__":
    test_collect_rows_preserves_batch_provenance_without_domain_policy()
    print("[json-batch-collection] general source metadata preservation: pass")
