from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_ranker_runtime import score_quality_rows_distilled
from quality_ranker_training import QualityRankerTrainingConfig, train_quality_ranker


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_embedding_artifact(root: Path) -> tuple[Path, list[dict]]:
    rows: list[dict] = []
    uids: list[str] = []
    vectors: list[list[float]] = []
    classes = (
        ("pass", (1.0, 0.0)),
        ("fail", (-1.0, 0.0)),
        ("abstain", (0.0, 1.0)),
    )
    for label, vector in classes:
        for index in range(30):
            uid = f"{label}-{index:02d}"
            rows.append({"uid": uid, "text": f"{label} payload {index}"})
            uids.append(uid)
            vectors.append([vector[0], vector[1] + index * 0.0001])
    rows.append({"uid": "ood", "text": "orthogonal outlier"})
    uids.append("ood")
    vectors.append([0.0, -1.0])
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    vectors_path = root / "embeddings.npz"
    np.savez(
        vectors_path,
        uids=np.asarray(uids),
        vectors=matrix,
        text_sha256s=np.asarray(
            [hashlib.sha256(row["text"].encode()).hexdigest() for row in rows]
        ),
    )
    manifest = {
        "schema_version": "semantic-embedding-artifact-v1",
        "provider_id": "fixture-encoder",
        "provider_identity_sha256": "e" * 64,
        "corpus_sha256": "c" * 64,
        "dimensions": 2,
        "record_count": len(uids),
        "vectors_file": vectors_path.name,
        "vectors_sha256": _sha256(vectors_path),
        "text_sha256_in_vectors": True,
    }
    manifest_path = root / "embedding_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, rows


def _write_observations(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if row["uid"] == "ood":
                continue
            label = row["uid"].split("-", 1)[0]
            payload = {
                "schema_version": "quality-teacher-corpus-observation-v3",
                "teacher_panel_sha256": "t" * 64,
                "quality_runtime_sha256": "r" * 64,
                "chunk_uid": row["uid"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "available_teacher_ids": ["one", "two", "three"],
                "unavailable_teacher_ids": [],
                "policy_results": [
                    {
                        "policy_id": policy_id,
                        "panel_decision": label,
                        "decision_source": "teacher_panel",
                        "decision_reason_codes": [f"fixture_{label}"],
                        "first_pass": [],
                        "second_pass": None,
                    }
                    for policy_id in POLICY_IDS
                ],
            }
            handle.write(json.dumps(payload) + "\n")


def test_training_artifact_drives_fast_scoring_and_ood_retention() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        labeled_rows = [row for row in rows if row["uid"] != "ood"]
        calibration_rows = [
            row for row in labeled_rows if int(row["uid"].rsplit("-", 1)[1]) < 20
        ]
        protected_rows = [row for row in labeled_rows if row not in calibration_rows]
        calibration_observations = root / "calibration_observations.jsonl"
        protected_observations = root / "protected_observations.jsonl"
        _write_observations(calibration_observations, calibration_rows)
        _write_observations(protected_observations, protected_rows)
        ranker_manifest = train_quality_ranker(
            embedding_manifest_path=embedding_manifest,
            calibration_observation_paths=(calibration_observations,),
            protected_observation_paths=(protected_observations,),
            output_dir=root / "ranker",
            config=QualityRankerTrainingConfig(
                seed="fixture-ranker",
                minimum_class_examples=6,
                minimum_fail_predictions=1,
                minimum_test_negatives=2,
                normal_maximum_false_positive_rate=0.10,
                hard_maximum_false_positive_rate=0.20,
                minimum_decision_confidence=0.50,
                ood_quantile=0.01,
            ),
        )

        results, audit = score_quality_rows_distilled(
            rows,
            embedding_manifest_path=embedding_manifest,
            ranker_manifest_path=ranker_manifest,
        )

        assert results["fail-00"][2].decision.value == "fail"
        assert results["pass-00"][2].decision.value == "pass"
        assert all(result.out_of_distribution for result in results["ood"])
        assert audit["teacher_requests"] == 0
        assert audit["runtime_method"] == "distilled_quality_ranker_v1"
        assert audit["input_chunks"] == len(rows)


def test_missing_policy_support_materializes_abstain_only_head() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        labeled_rows = [row for row in rows if row["uid"] != "ood"]
        calibration_rows = [
            row for row in labeled_rows if int(row["uid"].rsplit("-", 1)[1]) < 20
        ]
        protected_rows = [row for row in labeled_rows if row not in calibration_rows]
        calibration_observations = root / "calibration_observations.jsonl"
        protected_observations = root / "protected_observations.jsonl"
        _write_observations(calibration_observations, calibration_rows)
        _write_observations(protected_observations, protected_rows)
        payloads = [
            json.loads(line) for line in calibration_observations.read_text().splitlines()
        ]
        for payload in payloads:
            payload["policy_results"] = [
                result
                for result in payload["policy_results"]
                if result["policy_id"] != "q1_correctness_evidence"
            ]
        calibration_observations.write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )
        ranker_manifest = train_quality_ranker(
            embedding_manifest_path=embedding_manifest,
            calibration_observation_paths=(calibration_observations,),
            protected_observation_paths=(protected_observations,),
            output_dir=root / "ranker",
            config=QualityRankerTrainingConfig(
                seed="fixture-ranker-missing-head",
                minimum_class_examples=6,
                minimum_fail_predictions=1,
                minimum_test_negatives=2,
                normal_maximum_false_positive_rate=0.10,
                hard_maximum_false_positive_rate=0.20,
                minimum_decision_confidence=0.50,
                ood_quantile=0.01,
            ),
        )

        results, audit = score_quality_rows_distilled(
            rows,
            embedding_manifest_path=embedding_manifest,
            ranker_manifest_path=ranker_manifest,
        )

        assert audit["policy_heads"] == list(POLICY_IDS)
        assert results["pass-00"][0].decision.value == "abstain"
        assert results["pass-00"][0].normal_failure_threshold is None
        assert results["pass-00"][0].hard_failure_threshold is None


def test_training_rejects_protected_uid_or_text_overlap() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        observations = root / "shared_observations.jsonl"
        _write_observations(observations, rows[:30])

        try:
            train_quality_ranker(
                embedding_manifest_path=embedding_manifest,
                calibration_observation_paths=(observations,),
                protected_observation_paths=(observations,),
                output_dir=root / "ranker",
                config=QualityRankerTrainingConfig(
                    seed="fixture-overlap",
                    minimum_class_examples=2,
                    minimum_fail_predictions=1,
                    minimum_test_negatives=2,
                    normal_maximum_false_positive_rate=0.10,
                    hard_maximum_false_positive_rate=0.20,
                    minimum_decision_confidence=0.50,
                    ood_quantile=0.01,
                ),
            )
        except RuntimeError as error:
            assert "protected_observation_overlap" in str(error)
        else:
            raise AssertionError("Protected observations must be independent")


def test_training_rejects_protected_runtime_identity_mismatch() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        calibration = root / "calibration.jsonl"
        protected = root / "protected.jsonl"
        _write_observations(calibration, rows[:30])
        _write_observations(protected, rows[30:60])
        protected.write_text(
            protected.read_text(encoding="utf-8").replace(
                '"quality_runtime_sha256": "' + "r" * 64 + '"',
                '"quality_runtime_sha256": "' + "s" * 64 + '"',
            ),
            encoding="utf-8",
        )

        try:
            train_quality_ranker(
                embedding_manifest_path=embedding_manifest,
                calibration_observation_paths=(calibration,),
                protected_observation_paths=(protected,),
                output_dir=root / "ranker",
                config=QualityRankerTrainingConfig(
                    seed="fixture-runtime-mismatch",
                    minimum_class_examples=2,
                    minimum_fail_predictions=1,
                    minimum_test_negatives=2,
                    normal_maximum_false_positive_rate=0.10,
                    hard_maximum_false_positive_rate=0.20,
                    minimum_decision_confidence=0.50,
                    ood_quantile=0.01,
                ),
            )
        except RuntimeError as error:
            assert "protected_observation_runtime_mismatch" in str(error)
        else:
            raise AssertionError("Protected observations must share the calibration runtime")


def test_training_rejects_observation_text_hash_mismatch() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        calibration = root / "calibration.jsonl"
        protected = root / "protected.jsonl"
        _write_observations(calibration, rows[:30])
        _write_observations(protected, rows[30:60])
        payloads = [json.loads(line) for line in calibration.read_text().splitlines()]
        payloads[0]["text_sha256"] = "0" * 64
        calibration.write_text(
            "".join(json.dumps(payload) + "\n" for payload in payloads),
            encoding="utf-8",
        )

        try:
            train_quality_ranker(
                embedding_manifest_path=embedding_manifest,
                calibration_observation_paths=(calibration,),
                protected_observation_paths=(protected,),
                output_dir=root / "ranker",
                config=QualityRankerTrainingConfig(
                    seed="fixture-text-mismatch",
                    minimum_class_examples=2,
                    minimum_fail_predictions=1,
                    minimum_test_negatives=2,
                    normal_maximum_false_positive_rate=0.10,
                    hard_maximum_false_positive_rate=0.20,
                    minimum_decision_confidence=0.50,
                    ood_quantile=0.01,
                ),
            )
        except RuntimeError as error:
            assert "teacher_observation_text_mismatch" in str(error)
        else:
            raise AssertionError("Observation text identity mismatch must fail")


def test_runtime_rejects_stale_embedding_for_changed_text() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        embedding_manifest, rows = _write_embedding_artifact(root)
        labeled_rows = [row for row in rows if row["uid"] != "ood"]
        calibration = root / "calibration.jsonl"
        protected = root / "protected.jsonl"
        _write_observations(calibration, labeled_rows[:60])
        _write_observations(protected, labeled_rows[60:])
        ranker_manifest = train_quality_ranker(
            embedding_manifest_path=embedding_manifest,
            calibration_observation_paths=(calibration,),
            protected_observation_paths=(protected,),
            output_dir=root / "ranker",
            config=QualityRankerTrainingConfig(
                seed="fixture-stale-runtime",
                minimum_class_examples=6,
                minimum_fail_predictions=1,
                minimum_test_negatives=2,
                normal_maximum_false_positive_rate=0.10,
                hard_maximum_false_positive_rate=0.20,
                minimum_decision_confidence=0.50,
                ood_quantile=0.01,
            ),
        )
        changed = [{**rows[0], "text": "changed after embedding"}]

        try:
            score_quality_rows_distilled(
                changed,
                embedding_manifest_path=embedding_manifest,
                ranker_manifest_path=ranker_manifest,
            )
        except RuntimeError as error:
            assert "Quality embedding text identity mismatch" in str(error)
        else:
            raise AssertionError("Runtime must reject stale embeddings")


if __name__ == "__main__":
    test_training_artifact_drives_fast_scoring_and_ood_retention()
    test_missing_policy_support_materializes_abstain_only_head()
    test_training_rejects_protected_uid_or_text_overlap()
    test_training_rejects_protected_runtime_identity_mismatch()
    test_training_rejects_observation_text_hash_mismatch()
    test_runtime_rejects_stale_embedding_for_changed_text()
    print("[quality-ranker-runtime-v1] frozen artifact inference and OOD retention: pass")
