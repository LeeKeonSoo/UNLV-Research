#!/usr/bin/env python3
"""Automated metamorphic checks for temporal-code Stage-B proxies."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_selection import local_stage_b_features, score_stage_b  # noqa: E402


def _row(uid: str, text: str) -> dict:
    return {
        "chunk_uid": uid,
        "split": "train",
        "stage_a_pass": True,
        "bundle_id": "fixture",
        "repository_identity": "fixture/repo",
        "path": f"src/{uid}.py",
        "change_type": "modified",
        "content_type": "code",
        "chunk_kind": "function",
        "text": text,
    }


def main() -> int:
    base = _row(
        "base",
        "def parse(value):\n    if value is None:\n        return []\n    return value.split(',')\n",
    )
    formatted = _row(
        "formatted",
        "def parse( value ):\n\n    if value is None:\n        return [ ]\n\n    return value.split( ',' )\n",
    )
    renamed = _row(
        "renamed",
        "def decode(payload):\n    if payload is None:\n        return []\n    return payload.split(',')\n",
    )
    commented = _row(
        "commented",
        "def parse(value):\n"
        "    # General information and flexible behavior for many caller situations.\n"
        "    # Additional details do not change executable semantics.\n"
        "    if value is None:\n"
        "        return []\n"
        "    return value.split(',')\n",
    )
    distinct = _row(
        "distinct",
        "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n",
    )
    features = {row["chunk_uid"]: local_stage_b_features(row) for row in [base, formatted, renamed, commented]}
    assert features["base"]["code_quality_proxy"] == features["formatted"]["code_quality_proxy"], features
    assert features["base"]["code_quality_proxy"] == features["renamed"]["code_quality_proxy"], features
    assert features["base"]["code_quality_proxy"] == features["commented"]["code_quality_proxy"], features
    assert features["base"]["semantic_token_proxy_count"] == features["commented"]["semantic_token_proxy_count"], features
    assert features["commented"]["token_proxy_count"] > features["base"]["token_proxy_count"], features

    single = score_stage_b([base], quality_weight=0.8, redundancy_weight=0.2)
    duplicated = score_stage_b([base, renamed, distinct], quality_weight=0.8, redundancy_weight=0.2)
    single_risk = single[0]["stage_b_evidence"]["soft_redundancy_risk"]
    by_id = {row["chunk_uid"]: row["stage_b_evidence"] for row in duplicated}
    assert single_risk == 0.0, single
    assert by_id["base"]["soft_redundancy_risk"] >= 0.85, duplicated
    assert by_id["renamed"]["soft_redundancy_risk"] >= 0.85, duplicated
    assert by_id["distinct"]["soft_structural_redundancy_risk"] == 0.0, duplicated
    print("[temporal-code-stage-b-metamorphic] formatting, rename, and comment-only quality invariance: pass")
    print("[temporal-code-stage-b-metamorphic] duplicate multiplication and distinct-structure response: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
