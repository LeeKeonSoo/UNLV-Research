from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_ranker_enrichment import (
    materialize_augmented_corpus,
    materialize_target_policy_observations,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_target_policy_filter_keeps_one_bound_label_per_fixture() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        fixture_path = root / "fixtures.jsonl"
        observation_path = root / "observations.jsonl"
        output_path = root / "target.jsonl"
        fixtures = [
            {
                "chunk_uid": "fixture-1",
                "uid": "fixture-1",
                "text": "payload one",
                "fixture_policy_id": "q2_semantic_coherence",
                "fixture_class": "fail",
                "expected_decision": "fail",
                "expected_reason_code": "internal_semantic_contradiction",
            }
        ]
        observations = [
            {
                "chunk_uid": "fixture-1",
                "text_sha256": hashlib.sha256(b"payload one").hexdigest(),
                "teacher_panel_sha256": "a" * 64,
                "quality_runtime_sha256": "b" * 64,
                "aggregation_strategy": "single_teacher_confirmed_fail",
                "available_teacher_ids": ["luna"],
                "unavailable_teacher_ids": [],
                "policy_results": [
                    {
                        "policy_id": "q1_correctness_evidence",
                        "panel_decision": "abstain",
                        "first_pass": [],
                        "second_pass": None,
                    },
                    {
                        "policy_id": "q2_semantic_coherence",
                        "panel_decision": "fail",
                        "first_pass": [
                            {"reason_codes": ["internal_semantic_contradiction"]}
                        ],
                        "second_pass": [
                            {"reason_codes": ["internal_semantic_contradiction"]}
                        ],
                    },
                ],
            }
        ]
        _write_jsonl(fixture_path, fixtures)
        _write_jsonl(observation_path, observations)

        audit_path = materialize_target_policy_observations(
            fixture_path,
            observation_path,
            output_path,
        )
        filtered = json.loads(output_path.read_text().strip())
        audit = json.loads(audit_path.read_text())

    assert len(filtered["policy_results"]) == 1
    assert filtered["policy_results"][0]["policy_id"] == "q2_semantic_coherence"
    assert audit["target_policy_decision_counts"]["q2_semantic_coherence"]["fail"] == 1
    assert audit["expected_decision_match_count"] == 1


def test_augmented_corpus_rejects_text_identity_overlap() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        corpus_path = root / "corpus.jsonl"
        enrichment_path = root / "enrichment.jsonl"
        output_path = root / "augmented.jsonl"
        _write_jsonl(corpus_path, [{"uid": "raw-1", "text": "same payload"}])
        _write_jsonl(
            enrichment_path,
            [{"uid": "fixture-1", "chunk_uid": "fixture-1", "text": "same payload"}],
        )

        try:
            materialize_augmented_corpus(corpus_path, enrichment_path, output_path)
        except RuntimeError as error:
            assert "augmented_corpus_text_overlap" in str(error)
        else:
            raise AssertionError("Augmented corpus text identities must be disjoint")


if __name__ == "__main__":
    test_target_policy_filter_keeps_one_bound_label_per_fixture()
    test_augmented_corpus_rejects_text_identity_overlap()
    print("[quality-ranker-enrichment-v1] target-only labels and disjoint corpus: pass")
