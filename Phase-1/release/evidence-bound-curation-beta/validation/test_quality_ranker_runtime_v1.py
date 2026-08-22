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


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_identity(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("ranker_artifact_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_frozen_artifacts(root: Path) -> tuple[Path, Path, list[dict[str, str]]]:
    rows = [
        {"uid": "pass", "text": "coherent substantive payload"},
        {"uid": "fail", "text": "broken empty payload"},
        {"uid": "ood", "text": "out of support payload"},
    ]
    vectors = np.asarray(((1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)), dtype=np.float32)
    embedding_arrays = root / "embeddings.npz"
    np.savez(
        embedding_arrays,
        uids=np.asarray([row["uid"] for row in rows]),
        vectors=vectors,
        text_sha256s=np.asarray(
            [hashlib.sha256(row["text"].encode()).hexdigest() for row in rows]
        ),
    )
    embedding_manifest = root / "embedding_manifest.json"
    embedding_manifest.write_text(
        json.dumps(
            {
                "schema_version": "semantic-embedding-artifact-v1",
                "provider_id": "fixture-encoder",
                "provider_identity_sha256": "e" * 64,
                "corpus_sha256": "c" * 64,
                "dimensions": 2,
                "record_count": len(rows),
                "vectors_file": embedding_arrays.name,
                "vectors_sha256": _sha256(embedding_arrays),
                "text_sha256_in_vectors": True,
            }
        ),
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {"support": vectors[:2]}
    heads: list[dict[str, object]] = []
    for index, policy_id in enumerate(POLICY_IDS):
        coefficient_key = f"head_{index}_coefficients"
        intercept_key = f"head_{index}_intercepts"
        arrays[coefficient_key] = np.asarray(((4.0, 0.0), (-4.0, 0.0), (0.0, 2.0)))
        arrays[intercept_key] = np.zeros(3)
        heads.append(
            {
                "policy_id": policy_id,
                "class_labels": ["pass", "fail", "abstain"],
                "coefficient_key": coefficient_key,
                "intercept_key": intercept_key,
                "normal_fail_threshold": 0.90,
                "hard_fail_threshold": 0.75,
                "minimum_decision_confidence": 0.60,
                "train_count": 60,
                "calibration_count": 20,
                "test_count": 20,
            }
        )
    ranker_arrays = root / "quality_ranker_arrays.npz"
    np.savez(ranker_arrays, **arrays)
    payload: dict[str, object] = {
        "schema_version": "quality-ranker-artifact-v1",
        "arrays_file": ranker_arrays.name,
        "arrays_sha256": _sha256(ranker_arrays),
        "teacher_panel_sha256": "t" * 64,
        "encoder_provider_id": "fixture-encoder",
        "encoder_provider_identity_sha256": "e" * 64,
        "embedding_corpus_sha256": "c" * 64,
        "dimensions": 2,
        "ood_similarity_threshold": 0.50,
        "support_vectors_key": "support",
        "heads": heads,
    }
    payload["ranker_artifact_sha256"] = _artifact_identity(payload)
    ranker_manifest = root / "quality_ranker_manifest.json"
    ranker_manifest.write_text(json.dumps(payload), encoding="utf-8")
    return embedding_manifest, ranker_manifest, rows


def test_frozen_artifact_drives_runtime_scoring_without_teacher_calls() -> None:
    with TemporaryDirectory() as directory:
        embedding_manifest, ranker_manifest, rows = _write_frozen_artifacts(Path(directory))
        results, audit = score_quality_rows_distilled(
            rows,
            embedding_manifest_path=embedding_manifest,
            ranker_manifest_path=ranker_manifest,
        )

        assert all(result.decision.value == "pass" for result in results["pass"])
        assert all(result.decision.value == "fail" for result in results["fail"])
        assert all(result.out_of_distribution for result in results["ood"])
        assert audit["teacher_requests"] == 0
        assert audit["policy_heads"] == list(POLICY_IDS)


def test_runtime_rejects_stale_embedding_for_changed_text() -> None:
    with TemporaryDirectory() as directory:
        embedding_manifest, ranker_manifest, rows = _write_frozen_artifacts(Path(directory))
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
    test_frozen_artifact_drives_runtime_scoring_without_teacher_calls()
    test_runtime_rejects_stale_embedding_for_changed_text()
    print("[quality-ranker-runtime-v1] frozen artifact inference and identity checks: pass")
