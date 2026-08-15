from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quality_ranker_artifact import (
    QualityRankerArtifact,
    QualityRankerHead,
    load_quality_ranker_artifact,
    sha256_file,
)
from quality_ranker_policy import DistilledPolicyContract, distilled_policy_result
from quality_teacher_runtime import PanelPolicyResult


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _row_uid(row: Mapping) -> str:
    uid = row.get("chunk_uid") or row.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("Distilled Quality rows require chunk_uid or uid")
    return uid


def _load_embeddings(
    path: Path,
) -> tuple[dict, tuple[str, ...], tuple[str, ...], NDArray[np.float64]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    vectors_path = path.parent / str(manifest["vectors_file"])
    if sha256_file(vectors_path) != manifest.get("vectors_sha256"):
        raise RuntimeError("Distilled Quality embedding hash mismatch")
    with np.load(vectors_path, allow_pickle=False) as stored:
        uids = tuple(str(uid) for uid in stored["uids"].tolist())
        vectors = np.asarray(stored["vectors"], dtype=np.float64)
        if "text_sha256s" not in stored.files:
            raise RuntimeError("Distilled Quality embedding text identities are missing")
        text_sha256s = tuple(str(value) for value in stored["text_sha256s"].tolist())
    if len(text_sha256s) != len(uids):
        raise RuntimeError("Distilled Quality embedding text identities are incomplete")
    return manifest, uids, text_sha256s, vectors


def _probabilities(
    artifact: QualityRankerArtifact,
    head: QualityRankerHead,
    vectors: NDArray[np.float64],
) -> NDArray[np.float64]:
    coefficients = artifact.arrays[head.coefficient_key]
    intercepts = artifact.arrays[head.intercept_key]
    logits = vectors @ coefficients.T + intercepts
    if coefficients.shape[0] == 1 and len(head.class_labels) == 2:
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        return np.column_stack((1.0 - positive, positive))
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def score_quality_rows_distilled(
    rows: Sequence[Mapping],
    *,
    embedding_manifest_path: Path,
    ranker_manifest_path: Path,
) -> tuple[dict[str, tuple[PanelPolicyResult, ...]], dict]:
    artifact = load_quality_ranker_artifact(ranker_manifest_path)
    observed_policy_ids = tuple(head.policy_id for head in artifact.heads)
    if observed_policy_ids != QUALITY_POLICY_IDS:
        raise RuntimeError("Quality ranker artifact must contain the ordered Q1-Q4 policy heads")
    embedding, embedding_uids, embedding_text_sha256s, all_vectors = _load_embeddings(
        embedding_manifest_path
    )
    if str(embedding["provider_id"]) != artifact.encoder_provider_id:
        raise RuntimeError("Quality ranker encoder provider mismatch")
    if str(embedding["provider_identity_sha256"]) != artifact.encoder_provider_identity_sha256:
        raise RuntimeError("Quality ranker encoder identity mismatch")
    if all_vectors.shape[1] != artifact.dimensions:
        raise RuntimeError("Quality ranker embedding dimensions mismatch")
    position = {uid: index for index, uid in enumerate(embedding_uids)}
    text_sha256_by_uid = dict(
        zip(embedding_uids, embedding_text_sha256s, strict=True)
    )
    row_uids = tuple(_row_uid(row) for row in rows)
    missing = sorted(set(row_uids) - set(position))
    if missing:
        raise RuntimeError(f"Quality embedding artifact misses {len(missing)} input chunks")
    text_mismatches = [
        uid
        for uid, row in zip(row_uids, rows, strict=True)
        if text_sha256_by_uid[uid]
        != hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
    ]
    if text_mismatches:
        raise RuntimeError(
            f"Quality embedding text identity mismatch for {len(text_mismatches)} input chunks"
        )
    vectors = all_vectors[[position[uid] for uid in row_uids]]
    supports = artifact.arrays[artifact.support_vectors_key]
    nearest_support = (vectors @ supports.T).max(axis=1)
    ood = nearest_support < artifact.ood_similarity_threshold
    probabilities = {
        head.policy_id: _probabilities(artifact, head, vectors)
        for head in artifact.heads
    }
    results: dict[str, tuple[PanelPolicyResult, ...]] = {}
    uncertain_units = 0
    for row_index, uid in enumerate(row_uids):
        policy_results: list[PanelPolicyResult] = []
        for head in artifact.heads:
            contract = DistilledPolicyContract(
                policy_id=head.policy_id,
                class_labels=head.class_labels,
                normal_fail_threshold=head.normal_fail_threshold,
                hard_fail_threshold=head.hard_fail_threshold,
                minimum_decision_confidence=head.minimum_decision_confidence,
                ranker_artifact_sha256=artifact.artifact_sha256,
            )
            result = distilled_policy_result(
                contract,
                class_probabilities=probabilities[head.policy_id][row_index],
                out_of_distribution=bool(ood[row_index]),
            )
            policy_results.append(result)
        uncertain_units += int(
            bool(ood[row_index])
            or any(
                "quality_ranker_low_confidence_abstain" in result.reason_codes
                for result in policy_results
            )
        )
        results[uid] = tuple(policy_results)
    return results, {
        "runtime_method": "distilled_quality_ranker_v1",
        "ranker_manifest_path": str(ranker_manifest_path),
        "ranker_artifact_sha256": artifact.artifact_sha256,
        "embedding_manifest_path": str(embedding_manifest_path),
        "embedding_manifest_sha256": sha256_file(embedding_manifest_path),
        "input_chunks": len(rows),
        "policy_heads": [head.policy_id for head in artifact.heads],
        "ood_chunks": int(np.count_nonzero(ood)),
        "uncertain_chunks": uncertain_units,
        "teacher_requests": 0,
        "uncertain_action": "not_select_unless_coverage_veto",
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "source_reputation_read": False,
        "domain_quota_read": False,
    }
