from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class QualityRankerArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class QualityRankerHead:
    policy_id: str
    class_labels: tuple[str, ...]
    coefficient_key: str
    intercept_key: str
    normal_fail_threshold: float | None
    hard_fail_threshold: float | None
    minimum_decision_confidence: float
    train_count: int
    calibration_count: int
    test_count: int


@dataclass(frozen=True, slots=True)
class QualityRankerArtifact:
    artifact_sha256: str
    teacher_panel_sha256: str
    encoder_provider_id: str
    encoder_provider_identity_sha256: str
    embedding_corpus_sha256: str
    dimensions: int
    ood_similarity_threshold: float
    support_vectors_key: str
    heads: tuple[QualityRankerHead, ...]
    arrays: Mapping[str, NDArray[np.float64]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(payload: dict) -> str:
    canonical = dict(payload)
    canonical.pop("ranker_artifact_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_quality_ranker_artifact(
    *,
    output_dir: Path,
    manifest: dict,
    arrays: Mapping[str, NDArray[np.float64]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "quality_ranker_arrays.npz"
    np.savez(
        arrays_path,
        **{key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
    )
    payload = dict(manifest)
    payload["schema_version"] = "quality-ranker-artifact-v1"
    payload["arrays_file"] = arrays_path.name
    payload["arrays_sha256"] = sha256_file(arrays_path)
    payload["ranker_artifact_sha256"] = _artifact_identity(payload)
    manifest_path = output_dir / "quality_ranker_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_quality_ranker_artifact(path: Path) -> QualityRankerArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "quality-ranker-artifact-v1":
        raise QualityRankerArtifactError("Unsupported Quality ranker artifact schema")
    identity = str(payload.get("ranker_artifact_sha256") or "")
    if identity != _artifact_identity(payload):
        raise QualityRankerArtifactError("Quality ranker manifest identity mismatch")
    arrays_path = path.parent / str(payload["arrays_file"])
    if sha256_file(arrays_path) != payload.get("arrays_sha256"):
        raise QualityRankerArtifactError("Quality ranker array identity mismatch")
    with np.load(arrays_path, allow_pickle=False) as stored:
        arrays = {
            key: np.asarray(stored[key], dtype=np.float64)
            for key in stored.files
        }
    heads = tuple(
        QualityRankerHead(
            policy_id=str(item["policy_id"]),
            class_labels=tuple(str(label) for label in item["class_labels"]),
            coefficient_key=str(item["coefficient_key"]),
            intercept_key=str(item["intercept_key"]),
            normal_fail_threshold=(
                None
                if item.get("normal_fail_threshold") is None
                else float(item["normal_fail_threshold"])
            ),
            hard_fail_threshold=(
                None
                if item.get("hard_fail_threshold") is None
                else float(item["hard_fail_threshold"])
            ),
            minimum_decision_confidence=float(item["minimum_decision_confidence"]),
            train_count=int(item["train_count"]),
            calibration_count=int(item["calibration_count"]),
            test_count=int(item["test_count"]),
        )
        for item in payload["heads"]
    )
    return QualityRankerArtifact(
        artifact_sha256=identity,
        teacher_panel_sha256=str(payload["teacher_panel_sha256"]),
        encoder_provider_id=str(payload["encoder_provider_id"]),
        encoder_provider_identity_sha256=str(payload["encoder_provider_identity_sha256"]),
        embedding_corpus_sha256=str(payload["embedding_corpus_sha256"]),
        dimensions=int(payload["dimensions"]),
        ood_similarity_threshold=float(payload["ood_similarity_threshold"]),
        support_vectors_key=str(payload["support_vectors_key"]),
        heads=heads,
        arrays=arrays,
    )
