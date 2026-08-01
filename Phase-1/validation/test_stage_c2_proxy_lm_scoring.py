from __future__ import annotations

from stage_c2_proxy_lm_scoring import read_jsonl_records, semantic_bucket


def test_semantic_bucket_is_deterministic_and_dimension_sensitive() -> None:
    # Given: two identical vectors and a vector with a different sign pattern.
    first = [1.0, -1.0, 0.5, -0.5]
    second = [1.0, -1.0, 0.5, -0.5]
    different = [-1.0, 1.0, -0.5, 0.5]

    # When: LSH bucket keys are calculated.
    first_bucket = semantic_bucket(first, prefix_bits=4)

    # Then: the index is stable and separates the opposing sign pattern.
    assert first_bucket == semantic_bucket(second, prefix_bits=4)
    assert first_bucket != semantic_bucket(different, prefix_bits=4)


def test_read_jsonl_records_repairs_literal_newline_inside_text(tmp_path) -> None:
    # Given: a legacy JSONL record split by an unescaped newline inside its text string.
    path = tmp_path / "legacy.jsonl"
    path.write_text('{"chunk_uid":"one","text":"first\nsecond"}\n', encoding="utf-8")

    # When: the frozen proxy boundary reads the record.
    rows, repaired = read_jsonl_records(path)

    # Then: the textual newline is preserved and its repair is auditable.
    assert rows == [{"chunk_uid": "one", "text": "first\nsecond"}]
    assert repaired == 1


if __name__ == "__main__":
    test_semantic_bucket_is_deterministic_and_dimension_sensitive()
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        test_read_jsonl_records_repairs_literal_newline_inside_text(Path(directory))
    print("[stage-c2-proxy-lm-scoring] semantic bucket: pass")
