from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Final

import numpy as np


JsonMap = dict[str, Any]
MODEL_RELATIVE_REDUNDANCY_REASON: Final = "model_relative_redundant_family_member"
SEMANTIC_ONLY_REASON: Final = "semantic_family_nonrepresentative_candidate"
PROXY_ONLY_REASON: Final = "proxy_evidence_low_learning_signal_candidate"
FORBIDDEN_EVIDENCE_FIELDS: Final = frozenset(
    {
        "quality",
        "quality_score",
        "human_quality_label",
        "utility",
        "nll",
        "benchmark",
        "benchmark_outcomes",
        "source",
        "source_identity",
        "domain",
        "target_retention_fraction",
        "budget",
    }
)


def _as_mapping(value: Any) -> JsonMap:
    return dict(value) if isinstance(value, dict) else {}


def _embedding(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(element, int | float) for element in value):
        return None
    vector = tuple(float(element) for element in value)
    return vector if any(element != 0.0 for element in vector) else None


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return -1.0
    numerator = sum(left_item * right_item for left_item, right_item in zip(left, right, strict=True))
    denominator = math.sqrt(sum(item * item for item in left)) * math.sqrt(sum(item * item for item in right))
    return numerator / denominator if denominator else -1.0


def _evidence(row: JsonMap) -> JsonMap:
    evidence = _as_mapping(row.get("stage_c2_proxy_evidence"))
    forbidden = sorted(FORBIDDEN_EVIDENCE_FIELDS.intersection(evidence))
    if forbidden:
        raise RuntimeError(f"Stage C-2 proxy evidence contains forbidden policy inputs: {', '.join(forbidden)}")
    return evidence


def _number(evidence: JsonMap, field: str) -> float | None:
    value = evidence.get(field)
    return float(value) if isinstance(value, int | float) else None


def _is_evaluable(evidence: JsonMap) -> bool:
    return (
        isinstance(evidence.get("semantic_bucket"), str)
        and bool(evidence["semantic_bucket"])
        and _embedding(evidence.get("embedding")) is not None
        and _number(evidence, "familiarity") is not None
        and _number(evidence, "novelty") is not None
        and _number(evidence, "gradient_alignment") is not None
    )


def _representative_key(row: JsonMap) -> tuple[float, float, float, str]:
    evidence = _evidence(row)
    return (
        -float(_number(evidence, "novelty") or 0.0),
        float(_number(evidence, "familiarity") or 1.0),
        -float(_number(evidence, "gradient_alignment") or 0.0),
        str(row["chunk_uid"]),
    )


def _thresholds(config: JsonMap) -> tuple[float, float, float, float]:
    index = _as_mapping(config.get("semantic_index"))
    evidence = _as_mapping(config.get("evidence_thresholds"))
    cosine_threshold = float(index.get("cosine_threshold", 0.98))
    minimum_familiarity = float(evidence.get("minimum_familiarity", 0.80))
    maximum_novelty = float(evidence.get("maximum_novelty", 0.20))
    maximum_gradient_alignment = float(evidence.get("maximum_gradient_alignment", 0.05))
    if not 0.0 < cosine_threshold <= 1.0:
        raise RuntimeError("Stage C-2 semantic cosine threshold must be in (0, 1]")
    return cosine_threshold, minimum_familiarity, maximum_novelty, maximum_gradient_alignment


def _ablation_mode(config: JsonMap) -> str:
    mode = str(config.get("ablation_mode", "joint"))
    if mode not in {"joint", "semantic_only", "proxy_only"}:
        raise RuntimeError("Stage C-2 ablation mode must be joint, semantic_only, or proxy_only")
    return mode


def select_model_relative_candidates(
    chunks: Iterable[JsonMap], config: JsonMap
) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    """Apply a candidate-only corpus-level selector from frozen proxy evidence."""
    cosine_threshold, minimum_familiarity, maximum_novelty, maximum_gradient_alignment = _thresholds(config)
    ablation_mode = _ablation_mode(config)
    evaluable: list[JsonMap] = []
    retained: list[JsonMap] = []
    for raw in sorted(chunks, key=lambda row: str(row["chunk_uid"])):
        row = dict(raw)
        evidence = _evidence(row)
        if _is_evaluable(evidence):
            evaluable.append(row)
            continue
        row["stage_c2_selection"] = {
            "candidate_evaluated": False,
            "decision": "retained_not_evaluated_missing_frozen_proxy_evidence",
            "runtime_authorization": "none_candidate_cannot_select_or_remove",
        }
        retained.append(row)

    rejected: list[JsonMap] = []
    if ablation_mode == "proxy_only":
        for candidate in evaluable:
            candidate_evidence = _evidence(candidate)
            familiar = float(_number(candidate_evidence, "familiarity") or 0.0)
            novel = float(_number(candidate_evidence, "novelty") or 0.0)
            aligned = float(_number(candidate_evidence, "gradient_alignment") or 0.0)
            removable = familiar >= minimum_familiarity and novel <= maximum_novelty and aligned <= maximum_gradient_alignment
            candidate["stage_c2_selection"] = {
                "candidate_evaluated": True,
                "accepted": not removable,
                "decision": "retained_proxy_only_candidate" if not removable else "removed_proxy_only_candidate",
                "removed_reason": PROXY_ONLY_REASON if removable else None,
                "runtime_authorization": "none_candidate_cannot_select_or_remove",
            }
            (rejected if removable else retained).append(candidate)
        return retained, rejected, {
            "schema_version": "stage-c2-model-relative-candidate-v1",
            "status": "candidate_only_development_artifact",
            "ablation_mode": ablation_mode,
            "runtime_authorization": "none_candidate_cannot_select_or_remove",
            "candidate_removed_chunks": len(rejected),
            "not_evaluated_chunks": sum(not row["stage_c2_selection"].get("candidate_evaluated", False) for row in retained),
            "policy_inputs": ["chunk_text", "frozen_proxy_evidence"],
            "forbidden_policy_inputs": sorted(FORBIDDEN_EVIDENCE_FIELDS),
            "semantic_index": {"cosine_threshold": cosine_threshold},
            "evidence_thresholds": {"minimum_familiarity": minimum_familiarity, "maximum_novelty": maximum_novelty, "maximum_gradient_alignment": maximum_gradient_alignment},
        }
    buckets: dict[str, list[JsonMap]] = defaultdict(list)
    for row in evaluable:
        buckets[str(_evidence(row)["semantic_bucket"])].append(row)
    for bucket_rows in buckets.values():
        vectors = [(row, _embedding(_evidence(row).get("embedding"))) for row in bucket_rows]
        vector_matrix = np.asarray([vector for _, vector in vectors if vector is not None], dtype=np.float32)
        if not len(vector_matrix):
            continue
        norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
        normalized_matrix = vector_matrix / np.maximum(norms, 1e-12)
        similarities = normalized_matrix @ normalized_matrix.T
        processed: set[str] = set()
        for index, (row, vector) in enumerate(vectors):
            if vector is None or str(row["chunk_uid"]) in processed:
                continue
            family = [
                candidate for candidate_index, (candidate, candidate_vector) in enumerate(vectors)
                if candidate_vector is not None and similarities[index, candidate_index] >= cosine_threshold
            ]
            processed.update(str(candidate["chunk_uid"]) for candidate in family)
            representative = min(family, key=_representative_key)
            for candidate in family:
                if candidate.get("stage_c2_selection"):
                    continue
                candidate_evidence = _evidence(candidate)
                familiar = float(_number(candidate_evidence, "familiarity") or 0.0)
                novel = float(_number(candidate_evidence, "novelty") or 0.0)
                aligned = float(_number(candidate_evidence, "gradient_alignment") or 0.0)
                removable = candidate is not representative and (
                    ablation_mode == "semantic_only"
                    or (familiar >= minimum_familiarity and novel <= maximum_novelty and aligned <= maximum_gradient_alignment)
                )
                if removable:
                    candidate["stage_c2_selection"] = {
                        "candidate_evaluated": True,
                        "accepted": False,
                        "removed_reason": SEMANTIC_ONLY_REASON if ablation_mode == "semantic_only" else MODEL_RELATIVE_REDUNDANCY_REASON,
                        "representative_chunk_uid": str(representative["chunk_uid"]),
                        "evidence": {"semantic_neighbor": True, "familiarity": familiar, "novelty": novel, "gradient_alignment": aligned},
                        "runtime_authorization": "none_candidate_cannot_select_or_remove",
                    }
                    rejected.append(candidate)
                    continue
                candidate["stage_c2_selection"] = {
                    "candidate_evaluated": True,
                    "accepted": True,
                    "decision": "retained_model_relative_candidate",
                    "representative_chunk_uid": str(representative["chunk_uid"]),
                    "runtime_authorization": "none_candidate_cannot_select_or_remove",
                }
                retained.append(candidate)
    return retained, rejected, {
        "schema_version": "stage-c2-model-relative-candidate-v1",
        "status": "candidate_only_development_artifact",
        "ablation_mode": ablation_mode,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "candidate_removed_chunks": len(rejected),
        "not_evaluated_chunks": sum(not row["stage_c2_selection"].get("candidate_evaluated", False) for row in retained),
        "policy_inputs": ["chunk_text", "frozen_proxy_evidence", "frozen_corpus_index"],
        "forbidden_policy_inputs": sorted(FORBIDDEN_EVIDENCE_FIELDS),
        "semantic_index": {"cosine_threshold": cosine_threshold},
        "evidence_thresholds": {
            "minimum_familiarity": minimum_familiarity,
            "maximum_novelty": maximum_novelty,
            "maximum_gradient_alignment": maximum_gradient_alignment,
        },
    }
