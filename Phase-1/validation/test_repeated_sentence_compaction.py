#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repeated_sentence_compaction import (
    RepeatedSentenceSettings,
    compact_repeated_sentences,
)
from run_curation import materialize
from validation.test_curation_runtime import _all_pass_quality_scorer


SENTENCE = (
    "The framework should retain one stable representative for this repeated "
    "training sentence."
)


def test_ten_exact_occurrences_keep_one_and_emit_reversible_traces() -> None:
    result = compact_repeated_sentences(
        ({"chunk_uid": "fixture::0000", "text": " ".join([SENTENCE] * 10)},),
        RepeatedSentenceSettings(
            minimum_occurrences=3,
            minimum_lexical_tokens=12,
            minimum_residual_chars=40,
        ),
    )

    assert result.records[0]["text"] == SENTENCE
    assert len(result.transformations) == 9
    assert {item["reason_code"] for item in result.transformations} == {
        "redundancy_intra_chunk_exact_sentence_repeat_compacted"
    }
    assert {item["representative_chunk_uid"] for item in result.transformations} == {
        "fixture::0000"
    }
    assert {item["representative_occurrence_index"] for item in result.transformations} == {0}
    assert [item["removed_occurrence_index"] for item in result.transformations] == list(
        range(1, 10)
    )


def test_two_occurrences_do_not_trigger_the_three_occurrence_boundary() -> None:
    text = f"{SENTENCE} {SENTENCE}"
    result = compact_repeated_sentences(
        ({"chunk_uid": "fixture::0000", "text": text},),
        RepeatedSentenceSettings(
            minimum_occurrences=3,
            minimum_lexical_tokens=12,
            minimum_residual_chars=40,
        ),
    )

    assert result.records[0]["text"] == text
    assert result.transformations == ()


def test_distinct_sentence_families_are_not_merged() -> None:
    variant = SENTENCE.replace("stable", "deterministic")
    text = " ".join([SENTENCE] * 3 + [variant] * 3)
    result = compact_repeated_sentences(
        ({"chunk_uid": "fixture::0000", "text": text},),
        RepeatedSentenceSettings(
            minimum_occurrences=3,
            minimum_lexical_tokens=12,
            minimum_residual_chars=40,
        ),
    )

    assert result.records[0]["text"] == f"{SENTENCE} {variant}"
    assert len(result.transformations) == 4
    assert len({item["span_sha256"] for item in result.transformations}) == 2


def test_compaction_is_blocked_when_the_survivor_violates_residual_boundary() -> None:
    text = " ".join([SENTENCE] * 10)
    result = compact_repeated_sentences(
        ({"chunk_uid": "fixture::0000", "text": text},),
        RepeatedSentenceSettings(
            minimum_occurrences=3,
            minimum_lexical_tokens=12,
            minimum_residual_chars=len(SENTENCE) + 1,
        ),
    )

    assert result.records[0]["text"] == text
    assert result.transformations == ()
    assert result.blocked_chunk_uids == ("fixture::0000",)


def _write_runtime_fixture(root: Path, mode: str) -> Path:
    input_path = root / f"input-{mode}.jsonl"
    input_path.write_text(
        json.dumps({"id": "repeated-document", "text": " ".join([SENTENCE] * 10)})
        + "\n",
        encoding="utf-8",
    )
    config_path = root / f"config-{mode}.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "curation-run-contract-v1",
                "status": "frozen_before_stage_a_b_c_materialization",
                "curation_mode": mode,
                "execution_scope": "development",
                "input": {
                    "candidate_files": [str(input_path)],
                    "text_fields": ["text"],
                    "defaults": {},
                },
                "output_dir": str(root / mode),
                "stage_b": {"max_chunk_chars": 6000},
                "stage_c": {
                    "minimum_residual_chars": 40,
                    "no_binding_budget_action": "selection_without_binding_budget",
                },
                "claim_boundary": "repeated-sentence-runtime-fixture",
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_full_runtime_materializes_one_sentence_in_framework_profile() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for mode in ("framework",):
            report = materialize(
                _write_runtime_fixture(root, mode),
                quality_scorer=_all_pass_quality_scorer,
            )
            rows = [
                json.loads(line)
                for line in (root / mode / "stage_c_curated_chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

            assert [row["text"] for row in rows] == [SENTENCE]
            assert report["summary"]["stage_b_repeated_sentence_transformations"] == 9
            impact = report["reason_code_impact_audit"]["stages"][
                "stage_b_span_transformation"
            ]["reasons"]["redundancy_intra_chunk_exact_sentence_repeat_compacted"]
            assert impact["chunks"] == 1
            assert impact["token_proxy_removed"] == 108
            assert report["coverage_impact_audit"]["residual_payload"]["passed"] is True


if __name__ == "__main__":
    test_ten_exact_occurrences_keep_one_and_emit_reversible_traces()
    test_two_occurrences_do_not_trigger_the_three_occurrence_boundary()
    test_distinct_sentence_families_are_not_merged()
    test_compaction_is_blocked_when_the_survivor_violates_residual_boundary()
    test_full_runtime_materializes_one_sentence_in_framework_profile()
    print("[repeated-sentence-compaction] exact intra-chunk compaction: pass")
