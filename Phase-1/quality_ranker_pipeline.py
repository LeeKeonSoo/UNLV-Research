#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quality_ranker_sampling import (
    CalibrationSampleConfig,
    normalized_text_sha256,
    select_calibration_rows,
    select_protected_rows,
)
from quality_ranker_training import QualityRankerTrainingConfig, train_quality_ranker
from semantic_embedding_artifact import PoolingMode
from semantic_embedding_runtime import EmbeddingDocument, EmbeddingProviderSpec, encode_documents


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    provider_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_path: Path
    revision: str = Field(min_length=1)
    pooling: PoolingMode
    max_length: int = Field(ge=8)
    batch_size: int = Field(ge=1)
    device: str = Field(min_length=1)
    cache_dir: Path
    append_eos: bool
    output_dir: Path


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_observation_paths: tuple[Path, ...] = Field(min_length=1)
    protected_observation_paths: tuple[Path, ...] = Field(min_length=1)
    output_dir: Path
    seed: str = Field(min_length=1)
    minimum_class_examples: int = Field(ge=1)
    minimum_fail_predictions: int = Field(ge=1)
    minimum_test_negatives: int = Field(ge=1)
    normal_maximum_false_positive_rate: float = Field(ge=0.0, lt=1.0)
    hard_maximum_false_positive_rate: float = Field(ge=0.0, lt=1.0)
    minimum_decision_confidence: float = Field(gt=0.0, le=1.0)
    ood_quantile: float = Field(ge=0.0, lt=0.5)


class QualityRankerProgramConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["quality-ranker-program-v1"]
    status: Literal["frozen_before_calibration"]
    corpus_path: Path
    calibration_sample_path: Path
    calibration_sample_size: int = Field(ge=64)
    calibration_seed: str = Field(min_length=1)
    protected_sample_path: Path
    protected_sample_size: int = Field(ge=64)
    protected_seed: str = Field(min_length=1)
    embedding: EmbeddingConfig
    training: TrainingConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_program(path: Path) -> QualityRankerProgramConfig:
    return QualityRankerProgramConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _validate_existing_calibration(
    corpus_rows: list[dict],
    calibration_rows: list[dict],
    expected_count: int,
) -> None:
    if len(calibration_rows) != expected_count:
        raise RuntimeError("Existing calibration sample count does not match frozen config")
    corpus_by_uid = {
        str(row.get("uid") or row.get("chunk_uid")): normalized_text_sha256(str(row["text"]))
        for row in corpus_rows
    }
    mismatches = [
        str(row.get("chunk_uid") or row.get("uid"))
        for row in calibration_rows
        if corpus_by_uid.get(str(row.get("chunk_uid") or row.get("uid")))
        != normalized_text_sha256(str(row["text"]))
    ]
    if mismatches:
        raise RuntimeError("Existing calibration sample is not an exact corpus subset")


def _write_jsonl(path: Path, rows: tuple[dict, ...] | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def materialize_sample(config: QualityRankerProgramConfig) -> Path:
    rows = _read_jsonl(config.corpus_path)
    if config.calibration_sample_path.is_file():
        calibration = _read_jsonl(config.calibration_sample_path)
        _validate_existing_calibration(rows, calibration, config.calibration_sample_size)
    else:
        calibration = list(
            select_calibration_rows(
                rows,
                CalibrationSampleConfig(
                    target_size=config.calibration_sample_size,
                    seed=config.calibration_seed,
                ),
            )
        )
        _write_jsonl(config.calibration_sample_path, calibration)
    protected = select_protected_rows(
        rows,
        calibration_rows=calibration,
        config=CalibrationSampleConfig(
            target_size=config.protected_sample_size,
            seed=config.protected_seed,
        ),
    )
    _write_jsonl(config.protected_sample_path, protected)
    calibration_uids = {str(row["chunk_uid"]) for row in calibration}
    protected_uids = {str(row["chunk_uid"]) for row in protected}
    calibration_hashes = {normalized_text_sha256(str(row["text"])) for row in calibration}
    protected_hashes = {normalized_text_sha256(str(row["text"])) for row in protected}
    audit = {
        "schema_version": "quality-ranker-sampling-split-audit-v2",
        "calibration_sample_path": str(config.calibration_sample_path),
        "calibration_sample_sha256": _sha256(config.calibration_sample_path),
        "calibration_sample_count": len(calibration),
        "protected_sample_path": str(config.protected_sample_path),
        "protected_sample_sha256": _sha256(config.protected_sample_path),
        "protected_sample_count": len(protected),
        "uid_overlap_count": len(calibration_uids & protected_uids),
        "normalized_text_overlap_count": len(calibration_hashes & protected_hashes),
        "corpus_path": str(config.corpus_path),
        "corpus_sha256": _sha256(config.corpus_path),
        "sampling_authority": "teacher_calibration_and_disjoint_protected_verification_only",
        "source_reputation_read": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
    }
    if audit["uid_overlap_count"] or audit["normalized_text_overlap_count"]:
        raise RuntimeError("Calibration and protected samples are not disjoint")
    audit_path = config.protected_sample_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return audit_path


def materialize_embeddings(config: QualityRankerProgramConfig) -> Path:
    rows = _read_jsonl(config.corpus_path)
    embedding = config.embedding
    return encode_documents(
        EmbeddingProviderSpec(
            provider_id=embedding.provider_id,
            provider_identity_sha256=embedding.provider_identity_sha256,
            model_id=str(embedding.model_path),
            revision=embedding.revision,
            pooling=embedding.pooling,
            max_length=embedding.max_length,
            batch_size=embedding.batch_size,
            device=embedding.device,
            cache_dir=embedding.cache_dir,
            append_eos=embedding.append_eos,
            model_path_is_local=True,
        ),
        tuple(
            EmbeddingDocument(
                uid=str(row.get("uid") or row.get("chunk_uid")),
                text=str(row["text"]),
            )
            for row in rows
        ),
        _sha256(config.corpus_path),
        embedding.output_dir,
    )


def materialize_ranker(config: QualityRankerProgramConfig) -> Path:
    training = config.training
    return train_quality_ranker(
        embedding_manifest_path=config.embedding.output_dir / "embedding_manifest.json",
        calibration_observation_paths=training.calibration_observation_paths,
        protected_observation_paths=training.protected_observation_paths,
        output_dir=training.output_dir,
        config=QualityRankerTrainingConfig(
            seed=training.seed,
            minimum_class_examples=training.minimum_class_examples,
            minimum_fail_predictions=training.minimum_fail_predictions,
            minimum_test_negatives=training.minimum_test_negatives,
            normal_maximum_false_positive_rate=training.normal_maximum_false_positive_rate,
            hard_maximum_false_positive_rate=training.hard_maximum_false_positive_rate,
            minimum_decision_confidence=training.minimum_decision_confidence,
            ood_quantile=training.ood_quantile,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the distilled Quality ranker artifacts.")
    parser.add_argument("action", choices=("sample", "embed", "train"))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_program(args.config)
    if args.action == "sample":
        output = materialize_sample(config)
    elif args.action == "embed":
        output = materialize_embeddings(config)
    else:
        output = materialize_ranker(config)
    print(json.dumps({"status": "complete", "action": args.action, "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
