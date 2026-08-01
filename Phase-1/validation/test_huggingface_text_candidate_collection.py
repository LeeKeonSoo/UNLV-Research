#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from collect_huggingface_text_candidate_pool import collect_rows


def test_collect_rows_preserves_declared_source_facts_and_stops_at_token_limit() -> None:
    # Given: source-declared rights facts and two upstream text records.
    source = {"source_name": "fixture/math", "source_uri": "https://example.invalid/dataset", "collected_at": "2026-07-27T00:00:00+09:00", "rights": {"status": "allowed", "license": "ODC-By-1.0"}, "pii_context": "general", "language": {"code": "en", "confidence": None}, "partition": {"source_tier": "declared_math_web", "content_type": "general_text"}}
    upstream = iter([{"text": "one two three", "url": "https://example.invalid/one", "date": "2024-01-01"}, {"text": "four five six seven", "url": "https://example.invalid/two", "date": "2024-01-02"}])

    # When: collection reaches the declared whitespace-token proxy limit.
    rows = collect_rows(upstream, source, token_limit=3)

    # Then: the source contract remains visible and no extra source row is collected.
    assert len(rows) == 1
    assert rows[0]["rights"] == {"status": "allowed", "license": "ODC-By-1.0"}
    assert rows[0]["provenance"]["source_name"] == "fixture/math"
    assert rows[0]["partition"]["source_document_uri"] == "https://example.invalid/one"
    assert rows[0]["partition"]["source_document_date"] == "2024-01-01"
    assert rows[0]["token_proxy"] == 3
    assert rows[0].get("artifact_context") is None


def test_collect_rows_excludes_development_ids_and_requires_declared_license() -> None:
    # Given: a code source with a stable upstream ID and source-declared license allowlist.
    source = {
        "source_name": "fixture/code",
        "source_uri": "https://example.invalid/dataset",
        "collected_at": "2026-07-27T00:00:00+09:00",
        "rights": {"status": "allowed", "license": "per-record permissive allowlist"},
        "pii_context": "repository_code",
        "text_field": "content",
        "stable_record_id_field": "hexsha",
        "source_license_field": "licenses",
        "allowed_source_licenses": ["MIT"],
        "collection_admission": {"excluded_path_fragments": ["/tests/"], "minimum_bytes": 8, "maximum_bytes": 128},
        "path_field": "path",
        "partition": {"source_dataset": "fixture/code", "content_type": "code"},
    }
    upstream = iter(
        [
            {"content": "excluded record", "hexsha": "already-used", "licenses": ["MIT"]},
            {"content": "restricted record", "hexsha": "restricted", "licenses": ["GPL-3.0-only"]},
            {"content": "test-only", "hexsha": "test", "licenses": ["MIT"], "path": "src/tests/test_code.py"},
            {"content": "retain this independent record", "hexsha": "confirmatory", "licenses": ["MIT"]},
        ]
    )

    # When: collection receives the frozen development ID exclusion set.
    rows = collect_rows(upstream, source, token_limit=4, excluded_record_ids={"fixture/code::already-used"})

    # Then: only the source-licensed, source-disjoint record is materialized.
    assert [row["record_id"] for row in rows] == ["fixture/code::confirmatory"]
    assert rows[0]["rights"] == {"status": "allowed", "license": "MIT"}


def test_collect_rows_uses_declared_exact_token_counter() -> None:
    # Given: a collector source and an exact tokenizer-compatible counter.
    source = {
        "source_name": "fixture/exact",
        "source_uri": "https://example.invalid/exact",
        "collected_at": "2026-07-31T00:00:00Z",
        "rights": {"status": "allowed", "license": "fixture"},
    }
    upstream = iter([{"text": "first record"}, {"text": "second record"}])

    # When: the first whole record reaches the exact-token limit.
    rows = collect_rows(upstream, source, token_limit=50, token_counter=lambda text: len(text) * 10)

    # Then: collection and the stored count both use the declared counter.
    assert len(rows) == 1
    assert rows[0]["token_proxy"] == 120
    assert rows[0]["token_count"] == 120


def test_collect_rows_preserves_row_declared_language_when_configured() -> None:
    # Given: a multilingual source whose authoritative language is declared per row.
    source = {
        "source_name": "fixture/multilingual-code",
        "source_uri": "https://example.invalid/code",
        "collected_at": "2026-07-31T00:00:00Z",
        "rights": {"status": "allowed", "license": "fixture"},
        "text_field": "content",
        "language_field": "language",
        "language": {"code": "und", "confidence": None},
        "record_shape": "complete_source",
    }
    upstream = iter([{"content": "def add(a, b):\n    return a + b\n", "language": "Python"}])

    # When: the generic collector materializes the source row.
    rows = collect_rows(upstream, source, token_limit=4)

    # Then: the normalized row keeps the declared language instead of replacing it with und.
    assert rows[0]["language"] == {
        "code": "python",
        "confidence": 1.0,
        "declaration": "source_row",
    }
    assert rows[0]["record_shape"] == "complete_source"


if __name__ == "__main__":
    test_collect_rows_preserves_declared_source_facts_and_stops_at_token_limit()
    test_collect_rows_excludes_development_ids_and_requires_declared_license()
    test_collect_rows_uses_declared_exact_token_counter()
    test_collect_rows_preserves_row_declared_language_when_configured()
    print("[huggingface-text-collection] source-preserving token-limited collection: pass")
