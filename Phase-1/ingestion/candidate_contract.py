"""Validate the Stage-0 candidate record and quarantine contract."""

from __future__ import annotations

from typing import Any, Dict, List


CANDIDATE_RECORD_SCHEMA_VERSION = "candidate-corpus-record-v1"
ALLOWED_QUARANTINE_STATUSES = {"release_candidate", "quarantined", "rejected"}
ALLOWED_RIGHTS_STATUSES = {"allowed", "restricted", "unknown"}
ALLOWED_ARTIFACT_GENERATION = {"authored", "generated", "unknown"}


def validate_candidate_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if record.get("schema_version") != CANDIDATE_RECORD_SCHEMA_VERSION:
        errors.append("schema_version")
    for field in ("record_id", "text"):
        if not isinstance(record.get(field), str) or not str(record.get(field)).strip():
            errors.append(field)

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance")
    else:
        for field in ("source_name", "source_uri", "collected_at"):
            if not isinstance(provenance.get(field), str) or not str(provenance.get(field)).strip():
                errors.append(f"provenance.{field}")

    language = record.get("language")
    if not isinstance(language, dict) or not isinstance(language.get("code"), str):
        errors.append("language.code")
    elif language.get("version") is not None and not isinstance(language.get("version"), str):
        errors.append("language.version")

    artifact_context = record.get("artifact_context")
    if artifact_context is not None:
        if not isinstance(artifact_context, dict):
            errors.append("artifact_context")
        else:
            generation = artifact_context.get("generation")
            dependency_copy = artifact_context.get("dependency_copy")
            if generation is not None and generation not in ALLOWED_ARTIFACT_GENERATION:
                errors.append("artifact_context.generation")
            if dependency_copy is not None and not isinstance(dependency_copy, bool):
                errors.append("artifact_context.dependency_copy")

    rights = record.get("rights")
    if not isinstance(rights, dict) or rights.get("status") not in ALLOWED_RIGHTS_STATUSES:
        errors.append("rights.status")

    hazards = record.get("hazards")
    if not isinstance(hazards, dict):
        errors.append("hazards")
    else:
        for field in ("pii_detected", "secret_detected", "benchmark_contamination", "poisoning_suspected"):
            if not isinstance(hazards.get(field), bool):
                errors.append(f"hazards.{field}")

    quarantine = record.get("quarantine")
    if not isinstance(quarantine, dict):
        errors.append("quarantine")
    else:
        status = quarantine.get("status")
        reasons = quarantine.get("reasons")
        if status not in ALLOWED_QUARANTINE_STATUSES:
            errors.append("quarantine.status")
        if not isinstance(reasons, list) or not all(isinstance(reason, str) and reason for reason in reasons):
            errors.append("quarantine.reasons")
        elif status != "release_candidate" and not reasons:
            errors.append("quarantine.reasons_required")

    transformations = record.get("transformations")
    if not isinstance(transformations, list):
        errors.append("transformations")
    return sorted(set(errors))


def release_eligibility(record: Dict[str, Any]) -> Dict[str, Any]:
    validation_errors = validate_candidate_record(record)
    text_only = record.get("stage_a_policy") == "text_only_v2"
    rights = record.get("rights") or {}
    hazards = record.get("hazards") or {}
    quarantine = record.get("quarantine") or {}
    blockers: List[str] = []
    if validation_errors:
        blockers.append("invalid_candidate_contract")
    if not text_only:
        if rights.get("status") != "allowed":
            blockers.append("rights_not_allowed")
        for field in ("pii_detected", "secret_detected", "benchmark_contamination", "poisoning_suspected"):
            if hazards.get(field) is True:
                blockers.append(field)
    if quarantine.get("status") != "release_candidate":
        blockers.append("not_release_candidate")
    if quarantine.get("reasons"):
        blockers.append("quarantine_reasons_present")
    return {
        "eligible": not blockers,
        "blockers": sorted(set(blockers)),
        "validation_errors": validation_errors,
    }
