from __future__ import annotations

from typing import Any


JsonMap = dict[str, Any]
REQUIRED_CORPORA = {"code_raw_like", "math_raw_like", "general_text_raw_like"}


def _mapping(value: Any, field: str) -> JsonMap:
    if not isinstance(value, dict):
        raise RuntimeError(f"Stage C-2 protocol artifact requires object field: {field}")
    return dict(value)


def build_development_protocol_report(artifacts: JsonMap) -> JsonMap:
    """Verify that three development corpora used one frozen Stage C-2 protocol."""
    if set(artifacts) != REQUIRED_CORPORA:
        raise RuntimeError("Stage C-2 report requires code_raw_like, math_raw_like, and general_text_raw_like")
    normalized: JsonMap = {}
    for corpus_id in sorted(REQUIRED_CORPORA):
        artifact = _mapping(artifacts[corpus_id], corpus_id)
        manifest = _mapping(artifact.get("manifest"), f"{corpus_id}.manifest")
        audit = _mapping(artifact.get("audit"), f"{corpus_id}.audit")
        scoring = _mapping(manifest.get("scoring"), f"{corpus_id}.manifest.scoring")
        thresholds = _mapping(audit.get("evidence_thresholds"), f"{corpus_id}.audit.evidence_thresholds")
        if manifest.get("status") != "frozen_proxy_evidence_ready":
            raise RuntimeError(f"Stage C-2 {corpus_id} manifest is not frozen-proxy ready")
        if audit.get("runtime_authorization") != "none_candidate_cannot_select_or_remove":
            raise RuntimeError(f"Stage C-2 {corpus_id} artifact has runtime authority")
        normalized[corpus_id] = {
            "model_id": manifest.get("model_id"),
            "model_sha256": manifest.get("model_sha256"),
            "input_records": manifest.get("input_records"),
            "max_length": scoring.get("max_length"),
            "semantic_index": scoring.get("semantic_index"),
            "thresholds": thresholds,
            "candidate_removed_chunks": audit.get("candidate_removed_chunks"),
            "not_evaluated_chunks": audit.get("not_evaluated_chunks"),
        }
    reference = normalized["code_raw_like"]
    fields = ("model_id", "model_sha256", "input_records", "max_length", "semantic_index", "thresholds")
    for corpus_id, details in normalized.items():
        mismatches = [field for field in fields if details[field] != reference[field]]
        if mismatches:
            raise RuntimeError(f"Stage C-2 frozen protocol mismatch for {corpus_id}: {', '.join(mismatches)}")
    return {
        "schema_version": "stage-c2-development-protocol-report-v1",
        "status": "development_protocol_verified_not_a_promotion_decision",
        "protocol_integrity_passed": True,
        "thresholds_identical_across_corpora": True,
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "corpora": normalized,
    }
