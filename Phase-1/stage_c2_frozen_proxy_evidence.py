from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final


JsonMap = dict[str, Any]
FORBIDDEN_FIELDS: Final = frozenset(
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
REQUIRED_MANIFEST_FIELDS: Final = ("model_id", "model_sha256", "calibration_snapshot_sha256")


def _require_number(row: JsonMap, field: str) -> float:
    value = row.get(field)
    if not isinstance(value, int | float):
        raise RuntimeError(f"Frozen proxy evidence requires numeric {field}")
    return float(value)


def _require_embedding(row: JsonMap) -> list[float]:
    value = row.get("embedding")
    if not isinstance(value, list) or not value or not all(isinstance(item, int | float) for item in value):
        raise RuntimeError("Frozen proxy evidence requires a non-empty numeric embedding")
    return [float(item) for item in value]


def _rank_fraction(values: list[float]) -> dict[float, float]:
    ordered = sorted(set(values))
    denominator = max(len(ordered) - 1, 1)
    return {value: index / denominator for index, value in enumerate(ordered)}


def build_frozen_proxy_evidence(
    proxy_rows: Iterable[JsonMap], proxy_manifest: JsonMap
) -> tuple[list[JsonMap], JsonMap]:
    """Seal frozen proxy outputs into selector-visible evidence without quality semantics."""
    missing_manifest_fields = [field for field in REQUIRED_MANIFEST_FIELDS if not proxy_manifest.get(field)]
    if missing_manifest_fields:
        raise RuntimeError(f"Frozen proxy manifest missing required fields: {', '.join(missing_manifest_fields)}")
    rows = [dict(row) for row in proxy_rows]
    chunk_uids: set[str] = set()
    proxy_nlls: list[float] = []
    for row in rows:
        forbidden = sorted(FORBIDDEN_FIELDS.intersection(row))
        if forbidden:
            raise RuntimeError(f"Frozen proxy evidence contains forbidden policy inputs: {', '.join(forbidden)}")
        chunk_uid = row.get("chunk_uid")
        semantic_bucket = row.get("semantic_bucket")
        if not isinstance(chunk_uid, str) or not chunk_uid:
            raise RuntimeError("Frozen proxy evidence requires a non-empty chunk_uid")
        if chunk_uid in chunk_uids:
            raise RuntimeError(f"Frozen proxy evidence contains duplicate chunk_uid: {chunk_uid}")
        if not isinstance(semantic_bucket, str) or not semantic_bucket:
            raise RuntimeError("Frozen proxy evidence requires a non-empty semantic_bucket")
        chunk_uids.add(chunk_uid)
        _require_embedding(row)
        proxy_nlls.append(_require_number(row, "proxy_nll"))
        _require_number(row, "gradient_alignment")
    ranks = _rank_fraction(proxy_nlls)
    evidence = [
        {
            "chunk_uid": str(row["chunk_uid"]),
            "semantic_bucket": str(row["semantic_bucket"]),
            "embedding": _require_embedding(row),
            "familiarity": 1.0 - ranks[_require_number(row, "proxy_nll")],
            "novelty": ranks[_require_number(row, "proxy_nll")],
            "gradient_alignment": _require_number(row, "gradient_alignment"),
        }
        for row in rows
    ]
    return evidence, {
        "schema_version": "stage-c2-frozen-proxy-evidence-v1",
        "status": "frozen_proxy_evidence_ready",
        "model_id": proxy_manifest["model_id"],
        "model_sha256": proxy_manifest["model_sha256"],
        "calibration_snapshot_sha256": proxy_manifest["calibration_snapshot_sha256"],
        "calibration": "within_frozen_snapshot_rank_of_proxy_nll",
        "selector_visible_fields": ["chunk_uid", "semantic_bucket", "embedding", "familiarity", "novelty", "gradient_alignment"],
        "forbidden_policy_inputs": sorted(FORBIDDEN_FIELDS),
    }
