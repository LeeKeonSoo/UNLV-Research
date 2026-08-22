from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


OBSERVATION_SCHEMA = "quality-teacher-corpus-observation-v3"
QUALITY_RUNTIME_MODULES = (
    "quality_fallback_evidence.py",
    "quality_model_evidence.py",
    "quality_operating_points.py",
    "quality_teacher_observation_codec.py",
)


def quality_runtime_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative_path in QUALITY_RUNTIME_MODULES:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def quality_task_id(
    panel_sha256: str,
    runtime_sha256: str,
    chunk_uid: str,
    text_sha256: str,
) -> str:
    payload = (
        f"{panel_sha256}\0{runtime_sha256}\0{chunk_uid}\0{text_sha256}\0combined_q1_q4"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def panel_result_to_mapping(result: Any) -> dict[str, object]:
    return {
        "policy_id": result.policy_id,
        "panel_decision": result.decision.value,
        "decision_source": result.decision_source,
        "decision_reason_codes": list(result.reason_codes),
        "class_probabilities": dict(result.class_probabilities),
        "failure_probability": result.failure_probability,
        "normal_failure_threshold": result.normal_failure_threshold,
        "hard_failure_threshold": result.hard_failure_threshold,
        "prediction_confidence": result.prediction_confidence,
        "out_of_distribution": result.out_of_distribution,
        "ranker_artifact_sha256": result.ranker_artifact_sha256,
        "first_pass": [vote.model_dump(mode="json") for vote in result.first_pass],
        "second_pass": (
            None
            if result.second_pass is None
            else [vote.model_dump(mode="json") for vote in result.second_pass]
        ),
    }


__all__ = [
    "OBSERVATION_SCHEMA",
    "panel_result_to_mapping",
    "quality_runtime_sha256",
    "quality_task_id",
]
