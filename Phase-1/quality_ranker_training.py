from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from quality_ranker_artifact import sha256_file, write_quality_ranker_artifact
from quality_ranker_policy import calibrate_failure_threshold
from quality_ranker_protected import (
    ProtectedObservationError,
    ProtectedThresholdConfig,
    ThresholdVerification,
    load_observation_universe,
    require_disjoint_observations,
    verify_threshold,
)


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


@dataclass(frozen=True, slots=True)
class QualityRankerTrainingConfig:
    seed: str
    minimum_class_examples: int
    minimum_fail_predictions: int
    minimum_test_negatives: int
    normal_maximum_false_positive_rate: float
    hard_maximum_false_positive_rate: float
    minimum_decision_confidence: float
    ood_quantile: float

    def __post_init__(self) -> None:
        if not self.seed or min(
            self.minimum_class_examples,
            self.minimum_fail_predictions,
            self.minimum_test_negatives,
        ) < 1:
            raise ValueError("Quality ranker training support must be positive")
        if not (
            0.0 <= self.normal_maximum_false_positive_rate
            <= self.hard_maximum_false_positive_rate
            < 1.0
        ):
            raise ValueError("Normal must have the stricter false-positive tolerance")
        if not 0.0 < self.minimum_decision_confidence <= 1.0:
            raise ValueError("Quality ranker confidence must be in (0, 1]")
        if not 0.0 <= self.ood_quantile < 0.5:
            raise ValueError("OOD quantile must be in [0, 0.5)")


def _load_embeddings(
    path: Path,
) -> tuple[dict, tuple[str, ...], tuple[str, ...], NDArray[np.float64]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    vectors_path = path.parent / str(manifest["vectors_file"])
    if sha256_file(vectors_path) != manifest.get("vectors_sha256"):
        raise RuntimeError("Embedding artifact hash mismatch")
    with np.load(vectors_path, allow_pickle=False) as stored:
        uids = tuple(str(uid) for uid in stored["uids"].tolist())
        vectors = np.asarray(stored["vectors"], dtype=np.float64)
        if "text_sha256s" not in stored.files:
            raise ProtectedObservationError("embedding_text_identity_missing")
        text_sha256s = tuple(str(value) for value in stored["text_sha256s"].tolist())
    if len(text_sha256s) != len(uids):
        raise ProtectedObservationError("embedding_text_identity_missing")
    return manifest, uids, text_sha256s, vectors


def _stable_partition(uids: list[str], seed: str) -> tuple[list[int], list[int]]:
    ordered = sorted(
        range(len(uids)),
        key=lambda index: hashlib.sha256(f"{seed}\0{uids[index]}".encode()).hexdigest(),
    )
    train_end = max(1, int(len(ordered) * 0.75))
    return ordered[:train_end], ordered[train_end:]


def _stratified_partitions(
    labels: list[str],
    uids: list[str],
    seed: str,
) -> tuple[list[int], list[int]]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    train: list[int] = []
    calibration: list[int] = []
    for label, indices in sorted(by_class.items()):
        local_uids = [uids[index] for index in indices]
        local_train, local_calibration = _stable_partition(
            local_uids, f"{seed}\0{label}"
        )
        train.extend(indices[index] for index in local_train)
        calibration.extend(indices[index] for index in local_calibration)
    return sorted(train), sorted(calibration)


def _predict_probabilities(
    model: LogisticRegression,
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(model.predict_proba(vectors), dtype=np.float64)


def _ood_threshold(vectors: NDArray[np.float64], quantile: float) -> float:
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.max(axis=1)
    return float(np.quantile(nearest, quantile))


def train_quality_ranker(
    *,
    embedding_manifest_path: Path,
    calibration_observation_paths: tuple[Path, ...],
    protected_observation_paths: tuple[Path, ...],
    output_dir: Path,
    config: QualityRankerTrainingConfig,
) -> Path:
    embedding, uids, text_sha256s, vectors = _load_embeddings(embedding_manifest_path)
    calibration_universe = load_observation_universe(calibration_observation_paths)
    protected_universe = load_observation_universe(protected_observation_paths)
    require_disjoint_observations(calibration_universe, protected_universe)
    position = {uid: index for index, uid in enumerate(uids)}
    text_sha256_by_uid = dict(zip(uids, text_sha256s, strict=True))
    calibration_by_uid = calibration_universe.by_uid()
    protected_by_uid = protected_universe.by_uid()
    missing_embedding_uids = (set(calibration_by_uid) | set(protected_by_uid)) - set(position)
    if missing_embedding_uids:
        raise ProtectedObservationError(
            "protected_observation_embedding_missing",
            len(missing_embedding_uids),
        )
    text_mismatches = {
        observation.chunk_uid
        for observation in (*calibration_universe.observations, *protected_universe.observations)
        if text_sha256_by_uid.get(observation.chunk_uid) != observation.text_sha256
    }
    if text_mismatches:
        raise ProtectedObservationError(
            "teacher_observation_text_mismatch",
            len(text_mismatches),
        )
    observed_uids = sorted(calibration_by_uid)
    protected_uids = sorted(protected_by_uid)
    policy_ids = list(QUALITY_POLICY_IDS)
    arrays: dict[str, NDArray[np.float64]] = {}
    heads: list[dict] = []
    support_indices: set[int] = set()
    for head_index, policy_id in enumerate(policy_ids):
        head_uids = [
            uid
            for uid in observed_uids
            if calibration_by_uid[uid].decision_for(policy_id) is not None
        ]
        head_labels = [str(calibration_by_uid[uid].decision_for(policy_id)) for uid in head_uids]
        protected_head_uids = [
            uid
            for uid in protected_uids
            if protected_by_uid[uid].decision_for(policy_id) is not None
        ]
        protected_labels = [
            str(protected_by_uid[uid].decision_for(policy_id)) for uid in protected_head_uids
        ]
        counts = Counter(head_labels)
        protected_counts = Counter(protected_labels)
        coefficient_key = f"head_{head_index}_coefficients"
        intercept_key = f"head_{head_index}_intercepts"
        if len(counts) < 2 or min(counts.values(), default=0) < config.minimum_class_examples:
            arrays[coefficient_key] = np.zeros((1, vectors.shape[1]), dtype=np.float64)
            arrays[intercept_key] = np.zeros(1, dtype=np.float64)
            heads.append(
                {
                    "policy_id": policy_id,
                    "class_labels": ["abstain"],
                    "coefficient_key": coefficient_key,
                    "intercept_key": intercept_key,
                    "normal_fail_threshold": None,
                    "hard_fail_threshold": None,
                    "minimum_decision_confidence": config.minimum_decision_confidence,
                    "train_count": 0,
                    "calibration_count": 0,
                    "test_count": len(protected_head_uids),
                    "label_counts": dict(sorted(counts.items())),
                    "protected_label_counts": dict(sorted(protected_counts.items())),
                    "training_status": "insufficient_support_abstain_only",
                }
            )
            continue
        head_vectors = vectors[[position[uid] for uid in head_uids]]
        train, calibration = _stratified_partitions(
            head_labels, head_uids, f"{config.seed}\0{policy_id}"
        )
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=0,
        ).fit(head_vectors[train], np.asarray(head_labels)[train])
        classes = tuple(str(label) for label in model.classes_)
        calibration_probabilities = _predict_probabilities(model, head_vectors[calibration])
        protected_vectors = vectors[[position[uid] for uid in protected_head_uids]]
        protected_probabilities = _predict_probabilities(model, protected_vectors)
        fail_index = classes.index("fail") if "fail" in classes else None
        if fail_index is None:
            normal_threshold = None
            hard_threshold = None
            normal_verification = ThresholdVerification(
                None, None, len(protected_head_uids), 0, None, "fail_class_missing"
            )
            hard_verification = normal_verification
        else:
            calibration_binary = np.asarray(
                [int(head_labels[index] == "fail") for index in calibration],
                dtype=np.int64,
            )
            protected_binary = np.asarray(
                [int(label == "fail") for label in protected_labels],
                dtype=np.int64,
            )
            normal_candidate = calibrate_failure_threshold(
                calibration_binary,
                calibration_probabilities[:, fail_index],
                maximum_false_positive_rate=config.normal_maximum_false_positive_rate,
                minimum_fail_predictions=config.minimum_fail_predictions,
            )
            hard_candidate = calibrate_failure_threshold(
                calibration_binary,
                calibration_probabilities[:, fail_index],
                maximum_false_positive_rate=config.hard_maximum_false_positive_rate,
                minimum_fail_predictions=config.minimum_fail_predictions,
            )
            normal_verification = verify_threshold(
                normal_candidate,
                protected_binary,
                protected_probabilities[:, fail_index],
                ProtectedThresholdConfig(
                    config.normal_maximum_false_positive_rate,
                    config.minimum_test_negatives,
                ),
            )
            hard_verification = verify_threshold(
                hard_candidate,
                protected_binary,
                protected_probabilities[:, fail_index],
                ProtectedThresholdConfig(
                    config.hard_maximum_false_positive_rate,
                    config.minimum_test_negatives,
                ),
            )
            normal_threshold = normal_verification.activated_threshold
            hard_threshold = hard_verification.activated_threshold
            if normal_threshold is not None and (
                hard_threshold is None or hard_threshold > normal_threshold
            ):
                fallback = verify_threshold(
                    normal_threshold,
                    protected_binary,
                    protected_probabilities[:, fail_index],
                    ProtectedThresholdConfig(
                        config.hard_maximum_false_positive_rate,
                        config.minimum_test_negatives,
                    ),
                )
                if fallback.activated_threshold is not None:
                    hard_threshold = normal_threshold
                    hard_verification = ThresholdVerification(
                        fallback.candidate_threshold,
                        fallback.activated_threshold,
                        fallback.negative_count,
                        fallback.false_positive_count,
                        fallback.false_positive_upper_bound,
                        "verified_normal_threshold_fallback",
                    )
        arrays[coefficient_key] = np.asarray(model.coef_, dtype=np.float64)
        arrays[intercept_key] = np.asarray(model.intercept_, dtype=np.float64)
        support_indices.update(position[head_uids[index]] for index in train)
        heads.append(
            {
                "policy_id": policy_id,
                "class_labels": list(classes),
                "coefficient_key": coefficient_key,
                "intercept_key": intercept_key,
                "normal_fail_threshold": normal_threshold,
                "hard_fail_threshold": hard_threshold,
                "minimum_decision_confidence": config.minimum_decision_confidence,
                "train_count": len(train),
                "calibration_count": len(calibration),
                "test_count": len(protected_head_uids),
                "label_counts": dict(sorted(counts.items())),
                "protected_label_counts": dict(sorted(protected_counts.items())),
                "normal_protected_verification": normal_verification.as_mapping(),
                "hard_protected_verification": hard_verification.as_mapping(),
                "training_status": "trained",
            }
        )
    if not support_indices:
        raise RuntimeError("Teacher observations cannot train any Quality policy head")
    support_vectors = vectors[sorted(support_indices)]
    arrays["ood_support_vectors"] = support_vectors
    manifest = {
        "lifecycle": "candidate_distilled_runtime",
        "teacher_panel_sha256": calibration_universe.panel_sha256,
        "teacher_runtime_sha256": calibration_universe.runtime_sha256,
        "teacher_aggregation_strategy": calibration_universe.aggregation_strategy,
        "encoder_provider_id": str(embedding["provider_id"]),
        "encoder_provider_identity_sha256": str(embedding["provider_identity_sha256"]),
        "embedding_corpus_sha256": str(embedding["corpus_sha256"]),
        "embedding_manifest_sha256": sha256_file(embedding_manifest_path),
        "dimensions": int(vectors.shape[1]),
        "ood_similarity_threshold": _ood_threshold(support_vectors, config.ood_quantile),
        "support_vectors_key": "ood_support_vectors",
        "calibration_observation_artifacts": [
            {"path": artifact.path, "sha256": artifact.sha256}
            for artifact in calibration_universe.artifacts
        ],
        "protected_observation_artifacts": [
            {"path": artifact.path, "sha256": artifact.sha256}
            for artifact in protected_universe.artifacts
        ],
        "protected_observation_count": len(protected_universe.observations),
        "protected_disjointness": "uid_and_text_sha256_passed",
        "threshold_verification": "one_sided_95_percent_wilson_upper_bound",
        "heads": heads,
        "forbidden_runtime_inputs_read": [],
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "source_reputation_read": False,
        "domain_quota_read": False,
    }
    return write_quality_ranker_artifact(
        output_dir=output_dir,
        manifest=manifest,
        arrays=arrays,
    )
