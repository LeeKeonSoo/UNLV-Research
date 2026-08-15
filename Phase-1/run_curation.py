#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from composition_audit import annotate_record, annotate_records, build_composition_audit
from composition_artifacts import (
    CompositionRecord,
    build_composition_artifacts,
    write_composition_artifacts,
)
from all_policy_stage_b import apply_quality_policy
from coverage_redundancy_bridge import coverage_families_from_redundancy_plan
from coverage_contract import CoverageExecutionScope
from curation_artifacts import load_json, save_json, sha256_file
from framework_objects import CoreId, StageId
from framework_runtime_bridge import (
    RuntimeStageRequest,
    RuntimeStageTicket,
    authorize_runtime_stage,
    build_foundation_report,
    load_runtime_foundation,
)
from general_web_span_compaction import build_plan as build_web_span_plan
from general_web_span_compaction import materialize_candidate_plan as materialize_web_span_plan
from ingestion.candidate_processing import process_candidate
from ingestion.input_adapter import adapt_raw_records
from model_provider_contract import load_provider_registry
from quality_rule_evidence import CANDIDATE_QUALITY_RULE_KEYS
from quality_operating_points import CurationMode, required_quality_pass_count
from quality_runtime_dispatch import (
    DistilledRuntimeConfig,
    QualityRuntimeRequest,
    TeacherOracleConfig,
    score_quality_runtime,
)
from reason_code_audit import build_reason_code_impact_audit
from redundancy_equivalence import RedundancyMode
from redundancy_checkpoint import load_or_build_redundancy
from redundancy_v2 import RedundancySettings
from stage_b_policy import STAGE_B_STRUCTURAL_POLICY_REASON_CODES, propose_stage_b_removals
from semantic_coverage_materializer import (
    materialize_semantic_coverage,
    validate_semantic_coverage_artifacts,
)


JsonMap = dict[str, Any]
STAGE_B_INVALID_CHUNK_REASON = "invalid_chunk_result"
STAGE_B_EXACT_DUPLICATE_REASON = "normalized_exact_duplicate"
STAGE_B_POLICY_REASON_CODES = {
    "stage_b_invalid_chunk": frozenset({STAGE_B_INVALID_CHUNK_REASON}),
    "stage_b_exact_duplicate": frozenset({STAGE_B_EXACT_DUPLICATE_REASON}),
    "stage_b_symmetric_near_duplicate": frozenset(
        {
            "redundancy_equivalent_family_nonrepresentative",
            "redundancy_contained_payload_nonrepresentative",
        }
    ),
    "stage_b_quality_distilled_ranker_v1": frozenset(
        {
            "quality_normal_qualified_fail",
            "quality_hard_qualified_fail",
            "quality_normal_retention_threshold_not_met",
            "quality_hard_retention_threshold_not_met",
        }
    ),
    **STAGE_B_STRUCTURAL_POLICY_REASON_CODES,
}
USER_FACING_MODE_PROFILES = {"normal": "normal_structural_v1", "hard": "hard_structural_v1"}
POLICY_FINGERPRINT_CONFIGS = (
    "configs/curation_framework_v1.json",
    "configs/framework_objects_v1.json",
    "configs/framework_profiles_v1.json",
    "configs/framework_runtime_bridge_v1.json",
    "configs/quality_teacher_luna_single_v1.json",
    "configs/quality_ranker_v1.json",
    "configs/redundancy_v2.json",
    "configs/curation_contract.json",
    "configs/core_policy_registry.json",
    "configs/policy_cards.json",
    "configs/policy_profiles.json",
)


@dataclass(frozen=True, slots=True)
class RuntimeWritePreflightError(RuntimeError):
    path: Path
    detail: str

    def __str__(self) -> str:
        return f"Runtime output path is not writable: {self.path} ({self.detail})"


def _probe_writable_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeWritePreflightError(path=path, detail=str(error)) from error
    probe = path / f".curation-write-probe-{os.getpid()}"
    try:
        with probe.open("x", encoding="ascii") as handle:
            handle.write("ok\n")
        probe.unlink()
    except OSError as error:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeWritePreflightError(path=path, detail=str(error)) from error


def preflight_runtime_writes(
    *, output_dir: Path, cache_path: Path, checkpoint_path: Path
) -> None:
    for path in dict.fromkeys((output_dir, cache_path.parent, checkpoint_path.parent)):
        _probe_writable_directory(path)


POLICY_FINGERPRINT_RUNTIME_MODULES = (
    "run_curation.py",
    "all_policy_stage_b.py",
    "composition_audit.py",
    "composition_artifacts.py",
    "content_router.py",
    "coverage_contract.py",
    "coverage_engine.py",
    "coverage_metrics.py",
    "coverage_rematerialization.py",
    "coverage_taxonomy.py",
    "curation_artifacts.py",
    "framework_objects.py",
    "framework_profiles.py",
    "framework_runtime_bridge.py",
    "model_provider_contract.py",
    "stage_permissions.py",
    "general_web_span_compaction.py",
    "ingestion/input_adapter.py",
    "ingestion/candidate_processing.py",
    "quality_decision_contract.py",
    "quality_rule_evidence.py",
    "quality_teacher_panel.py",
    "quality_teacher_json.py",
    "quality_teacher_response.py",
    "quality_teacher_runtime.py",
    "quality_teacher_materialization.py",
    "quality_teacher_batch_cache.py",
    "quality_teacher_unit_runtime.py",
    "quality_teacher_batch_runtime.py",
    "quality_teacher_adapters.py",
    "quality_ranker_artifact.py",
    "quality_ranker_policy.py",
    "quality_ranker_runtime.py",
    "quality_operating_points.py",
    "quality_stage_bridge.py",
    "quality_runtime_dispatch.py",
    "quality_retention.py",
    "reason_code_audit.py",
    "redundancy_checkpoint.py",
    "redundancy_equivalence.py",
    "redundancy_mode_policy.py",
    "redundancy_v2.py",
    "redundancy_v2_retrieval.py",
    "coverage_redundancy_bridge.py",
    "semantic_coverage_materializer.py",
    "semantic_coverage_bundle.py",
    "semantic_coverage_graph.py",
    "stage_b_policy.py",
    "stage_c_selection.py",
)


def load_config(path: Path) -> JsonMap:
    config = load_json(path)
    if config.get("status") != "frozen_before_stage_a_b_c_materialization":
        raise RuntimeError(f"Unexpected curation config status: {config.get('status')}")
    return config


def _policy_profile(profile_id: str) -> JsonMap:
    root = Path(__file__).resolve().parent
    payload = load_json(root / "configs" / "policy_profiles.json")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise RuntimeError("Policy profile registry requires a profiles list.")
    for profile in profiles:
        if isinstance(profile, dict) and profile.get("id") == profile_id:
            return profile
    raise RuntimeError(f"Unknown policy profile: {profile_id}")


def _effective_policy_digest(policy: JsonMap) -> str:
    encoded = json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_curation_mode(mode: str, *, execution_scope: str = "production") -> JsonMap:
    """Resolve a user-facing mode to one immutable complete runtime policy."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in USER_FACING_MODE_PROFILES:
        raise RuntimeError("Curation mode must be normal or hard")
    profile_id = USER_FACING_MODE_PROFILES[normalized_mode]
    profile = _policy_profile(profile_id)
    runtime_policy = profile.get("runtime_policy")
    if not isinstance(runtime_policy, dict):
        raise RuntimeError(f"Policy profile {profile_id} has no complete runtime_policy.")
    if execution_scope == "development":
        authorization = "development_candidate_release_blocked"
    elif execution_scope == "confirmatory":
        authorization = "confirmatory_candidate_release_blocked"
    else:
        raise RuntimeError(
            "Normal and Hard remain limited to development or confirmatory evaluation; production release is blocked."
        )
    effective_policy = json.loads(json.dumps(runtime_policy))
    return {
        "mode": normalized_mode,
        "profile_id": profile_id,
        "profile_status": profile.get("status"),
        "authorization": authorization,
        "enabled_policy_ids": list(profile.get("enabled_policy_ids") or []),
        "effective_policy": effective_policy,
        "effective_policy_sha256": _effective_policy_digest(effective_policy),
    }


def validate_run_policy_overrides(config: JsonMap, effective_policy: JsonMap) -> None:
    """Reject run-local switches that would mutate an immutable mode profile."""
    forbidden_sections = [name for name in ("stage_a", "stage_b_policy") if name in config]
    stage_b = config.get("stage_b") if isinstance(config.get("stage_b"), dict) else {}
    forbidden_stage_b = sorted(
        key
        for key in ("deduplicate_normalized_exact_text", "deduplicate_stage_a_text_exactly")
        if key in stage_b
    )
    if forbidden_sections or forbidden_stage_b:
        fields = forbidden_sections + [f"stage_b.{key}" for key in forbidden_stage_b]
        raise RuntimeError(
            "Run contract cannot override immutable profile policy fields: " + ", ".join(fields)
        )
    if not isinstance(effective_policy.get("stage_a"), dict):
        raise RuntimeError("Immutable profile is missing Stage-A policy.")
    if not isinstance(effective_policy.get("stage_b"), dict):
        raise RuntimeError("Immutable profile is missing Stage-B policy.")
    if not isinstance(effective_policy.get("stage_b_policy"), dict):
        raise RuntimeError("Immutable profile is missing Stage-B removal policy.")
    quality_runtime = effective_policy.get("quality_runtime")
    if not isinstance(quality_runtime, dict):
        raise RuntimeError("Immutable profile is missing the Quality runtime policy.")
    operating_point = CurationMode(str(quality_runtime.get("operating_point")))
    declared_pass_count = quality_runtime.get("minimum_passed_policies")
    expected_pass_count = required_quality_pass_count(operating_point)
    if declared_pass_count != expected_pass_count:
        raise RuntimeError(
            "Immutable profile Quality retention threshold differs from the frozen "
            f"operating point: expected {expected_pass_count}, observed {declared_pass_count}"
        )
    coverage = effective_policy.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("enforce_materialization_invariants") is not True:
        raise RuntimeError("Immutable profile must enforce Coverage materialization invariants.")


def validate_quality_candidate_scope(stage_b_policy: JsonMap, execution_scope: str) -> list[str]:
    artifact_settings = dict(stage_b_policy.get("structural_artifact_rules") or {})
    enabled = sorted(key for key in CANDIDATE_QUALITY_RULE_KEYS if artifact_settings.get(key) is True)
    span_settings = dict(stage_b_policy.get("quality_span_candidate_rules") or {})
    if span_settings.get("web_control_and_url_directory") is True:
        enabled.append("web_control_and_url_directory_span_candidate")
    if enabled and execution_scope != "development":
        raise RuntimeError(
            "Candidate Quality rules require execution_scope='development': " + ", ".join(enabled)
        )
    return enabled


def _stage_b_materialization_universe(
    passed: list[JsonMap], selected: list[JsonMap], not_selected: list[JsonMap]
) -> tuple[JsonMap, ...]:
    """Preserve Stage-B decisions while retaining the original chunk order."""
    decided = {str(row["chunk_uid"]): row for row in (*selected, *not_selected)}
    passed_ids = [str(row["chunk_uid"]) for row in passed]
    if len(decided) != len(passed) or set(decided) != set(passed_ids):
        raise RuntimeError("Stage-B decisions do not cover the complete Stage-B universe.")
    return tuple(decided[uid] for uid in passed_ids)


def _policy_fingerprint() -> JsonMap:
    """Hash the active policy declarations and runtime modules used by this run."""
    root = Path(__file__).resolve().parent
    return {
        "policy_configs": {path: sha256_file(root / path) for path in POLICY_FINGERPRINT_CONFIGS},
        "runtime_modules": {path: sha256_file(root / path) for path in POLICY_FINGERPRINT_RUNTIME_MODULES},
    }


def _read_jsonl(paths: Iterable[Path]) -> list[JsonMap]:
    rows: list[JsonMap] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", errors="replace") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _source_specs(input_config: JsonMap) -> list[JsonMap]:
    configured_sources = input_config.get("sources")
    if isinstance(configured_sources, list):
        return [source for source in configured_sources if isinstance(source, dict)]
    candidate_files = input_config.get("candidate_files")
    if isinstance(candidate_files, list):
        return [{"path": path, "text_fields": input_config.get("text_fields"), "defaults": input_config.get("defaults")} for path in candidate_files]
    raise RuntimeError("Input config requires a non-empty sources or candidate_files list.")


def _source_path(source: JsonMap) -> Path:
    path = source.get("path")
    if not isinstance(path, str) or not path.strip():
        raise RuntimeError("Each source requires a non-empty path.")
    return Path(path)


def _pretraining_audit(config: JsonMap, source_paths: list[Path]) -> JsonMap | None:
    audit_path_value = config.get("pretraining_audit_path")
    if not isinstance(audit_path_value, str) or not audit_path_value.strip():
        return None
    audit_path = Path(audit_path_value)
    audit = load_json(audit_path)
    audited_output = audit.get("audited_output")
    if not isinstance(audited_output, dict):
        raise RuntimeError("Pretraining audit requires an audited_output object.")
    audited_path = Path(str(audited_output.get("path") or ""))
    if audited_path not in source_paths:
        raise RuntimeError("Curation input must be the audited output declared by pretraining audit.")
    expected_sha = str(audited_output.get("sha256") or "")
    actual_sha = sha256_file(audited_path)
    if expected_sha != actual_sha:
        raise RuntimeError("Audited curation input hash does not match pretraining audit.")
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "status": audit.get("status"),
        "pretraining_eligible": audit.get("pretraining_eligible"),
        "audited_input_sha256": actual_sha,
    }


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _token_proxy(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, max_chunk_chars: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(line) > max_chunk_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(line[start : start + max_chunk_chars] for start in range(0, len(line), max_chunk_chars))
        elif current and len(current) + len(line) > max_chunk_chars:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]


def _audit_metadata(record: JsonMap) -> JsonMap:
    provenance = record["provenance"]
    rights = record["rights"]
    partition = record.get("partition")
    source_pool_role = partition.get("source_pool_role") if isinstance(partition, dict) else None
    return {
        "provenance": {
            "source_name": provenance["source_name"],
            "source_pool_role": source_pool_role,
            "license": rights["license"],
        },
        "composition": record["composition"],
    }


def _policy_metadata(record: JsonMap) -> JsonMap:
    """Expose only declared structural metadata that a Stage-C rule may inspect."""
    language = record.get("language") if isinstance(record.get("language"), dict) else {}
    partition = record.get("partition") if isinstance(record.get("partition"), dict) else {}
    artifact_context = record.get("artifact_context") if isinstance(record.get("artifact_context"), dict) else {}
    metadata = {
        "language_code": str(language.get("code") or "und"),
        "content_type": str(partition.get("content_type") or "unknown"),
    }
    path = partition.get("path")
    if isinstance(path, str) and path.strip():
        metadata["path"] = path.replace("\\", "/")
    if isinstance(language.get("version"), str) and language["version"].strip():
        metadata["language_version"] = language["version"]
    if artifact_context.get("generation") in {"authored", "generated", "unknown"}:
        metadata["declared_generation"] = artifact_context["generation"]
    if isinstance(artifact_context.get("dependency_copy"), bool):
        metadata["declared_dependency_copy"] = artifact_context["dependency_copy"]
    return metadata


def _stage_b_chunks(
    released: Iterable[JsonMap], stage_b_policy: JsonMap, *, text_only: bool = False
) -> tuple[list[JsonMap], list[JsonMap]]:
    maximum = int(stage_b_policy["max_chunk_chars"])
    exact_deduplication = stage_b_policy.get("deduplicate_stage_a_text_exactly") is True
    seen: dict[str, str] = {}
    passed: list[JsonMap] = []
    rejected: list[JsonMap] = []
    pending: list[tuple[str, str, JsonMap, str]] = []
    for record in released:
        for index, text in enumerate(chunk_text(str(record["text"]), maximum)):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk_uid = f"{record['record_id']}::{index:04d}"
            pending.append((digest, chunk_uid, record, text))
    for digest, chunk_uid, record, text in sorted(
        pending, key=lambda item: (item[0], item[1])
    ):
        visible = text.strip()
        reasons: list[str] = []
        is_invalid = not visible or all(character.isspace() or ord(character) < 32 for character in text)
        if is_invalid:
            reasons.append(STAGE_B_INVALID_CHUNK_REASON)
        representative_chunk_uid = seen.get(digest) if exact_deduplication and not is_invalid else None
        if representative_chunk_uid is not None:
            reasons.append(STAGE_B_EXACT_DUPLICATE_REASON)
        policy_metadata = {} if text_only else _policy_metadata(record)
        token_proxy = _token_proxy(text)
        chunk = {
            "chunk_uid": chunk_uid,
            "text": text,
            "token_proxy": token_proxy,
            "token_proxy_kind": "whitespace_proxy_non_training",
            "stage_a_record_id": record["record_id"],
            "stage_b_hard_gate_reasons": reasons,
            "stage_b_decision": {
                "accepted": not reasons,
                "trigger": "stage_b_hard_gate_reason_present" if reasons else "no_stage_b_hard_gate_reason",
                "non_trigger_boundary": "stage_a_text_digest_is_distinct" if not reasons else "declared_hard_gate_rejection",
                "reason_codes": reasons,
                "token_delta_proxy": -token_proxy if reasons else 0,
                "token_delta_proxy_kind": "whitespace_proxy_non_training",
                "utility_read": False,
                "benchmark_outcomes_read": False,
            },
            "audit_only_metadata": _audit_metadata(record),
            "stage_c_policy_metadata": policy_metadata,
            "composition": annotate_record({"text": text}),
            "stage_c_selector_visible": {
                "declared_language": not text_only,
                "declared_language_version": "language_version" in policy_metadata,
                "declared_content_type": not text_only,
                "declared_path": "path" in policy_metadata,
                "declared_artifact_context": bool(
                    policy_metadata.get("declared_generation")
                    or "declared_dependency_copy" in policy_metadata
                ),
                "source_name": False,
                "source_pool_role": False,
                "composition": False,
                "utility": False,
                "benchmark_outcomes": False,
            },
        }
        if representative_chunk_uid is not None:
            chunk["stage_b_decision"]["representative_chunk_uid"] = representative_chunk_uid
        if exact_deduplication and not is_invalid and digest not in seen:
            seen[digest] = chunk_uid
        if reasons:
            rejected.append(chunk)
        else:
            passed.append(chunk)
    return passed, rejected


def _coverage_impact_audit(
    *,
    passed: list[JsonMap],
    selected: list[JsonMap],
    rejected: list[JsonMap],
    not_selected: list[JsonMap],
    span_transformations: list[JsonMap],
    minimum_residual_chars: int,
    composition_audit: JsonMap,
) -> JsonMap:
    """Audit representative preservation without granting Coverage selection authority."""
    selected_ids = {str(row["chunk_uid"]) for row in selected}
    required_links: list[tuple[str, str | None, str]] = []
    for row in rejected:
        if STAGE_B_EXACT_DUPLICATE_REASON in row["stage_b_hard_gate_reasons"]:
            required_links.append(
                (
                    str(row["chunk_uid"]),
                    row["stage_b_decision"].get("representative_chunk_uid"),
                    STAGE_B_EXACT_DUPLICATE_REASON,
                )
            )
    for row in not_selected:
        selection = row.get("stage_b_policy") if isinstance(row.get("stage_b_policy"), dict) else {}
        reason = selection.get("removed_reason")
        redundancy_trace = (
            row.get("stage_b_redundancy_v2")
            if isinstance(row.get("stage_b_redundancy_v2"), dict)
            else {}
        )
        if reason in {"near_duplicate_representative_retained", "structural_scaffold_representative_retained"} or redundancy_trace.get("action") == "remove":
            required_links.append((str(row["chunk_uid"]), selection.get("representative_chunk_uid"), str(reason)))

    missing_links = [chunk_uid for chunk_uid, representative, _ in required_links if not representative]
    removal_reason_by_chunk_uid = {
        str(row["chunk_uid"]): str(row["stage_b_policy"].get("removed_reason"))
        for row in not_selected
    }
    removed_row_by_chunk_uid = {str(row["chunk_uid"]): row for row in not_selected}
    explicit_non_payload_reasons = {
        "explicit_generated_artifact",
        "license_comment_only_chunk",
        "empty_html_shell",
        "explicit_web_chrome_only_chunk",
        "explicit_error_navigation_only_chunk",
        "url_directory_only_chunk",
    }

    def is_authorized_terminal_non_selection(row: JsonMap) -> bool:
        selection = row.get("stage_b_policy")
        if not isinstance(selection, dict):
            return False
        if selection.get("removed_reason") in explicit_non_payload_reasons:
            return True
        failed_policy_ids = selection.get("failed_policy_ids")
        if isinstance(failed_policy_ids, list) and bool(failed_policy_ids):
            return True
        quality_decision = row.get("quality_stage_decision")
        if not isinstance(quality_decision, dict):
            return False
        passed_policy_ids = quality_decision.get("passed_policy_ids")
        required_pass_count = quality_decision.get("required_pass_count")
        return (
            quality_decision.get("stage_b_action") == "not_select"
            and quality_decision.get("stage_b_reason_code")
            in {
                "quality_normal_retention_threshold_not_met",
                "quality_hard_retention_threshold_not_met",
            }
            and isinstance(passed_policy_ids, list)
            and isinstance(required_pass_count, int)
            and len(passed_policy_ids) < required_pass_count
        )

    def resolve_representative_chain(representative: str) -> tuple[str, list[str], str | None]:
        chain: list[str] = []
        visited: set[str] = set()
        current = representative
        while True:
            if current in visited:
                return "cycle", [*chain, current], None
            visited.add(current)
            chain.append(current)
            if current in selected_ids:
                return "selected", chain, current
            removed_row = removed_row_by_chunk_uid.get(current)
            if removed_row is None:
                return "missing", chain, None
            selection = removed_row.get("stage_b_policy")
            if not isinstance(selection, dict):
                return "missing", chain, None
            reason = str(selection.get("removed_reason"))
            if is_authorized_terminal_non_selection(removed_row):
                return "non_payload", chain, current
            next_representative = selection.get("representative_chunk_uid")
            if not isinstance(next_representative, str) or not next_representative:
                return "missing", chain, None
            current = next_representative

    representative_resolutions = [
        (chunk_uid, representative, reason, *resolve_representative_chain(str(representative)))
        for chunk_uid, representative, reason in required_links
        if representative
    ]
    resolved_by_non_payload_removal = [
        {
            "chunk_uid": chunk_uid,
            "representative_chunk_uid": representative,
            "reason_code": reason,
            "representative_removed_reason": removal_reason_by_chunk_uid[str(terminal)],
        }
        for chunk_uid, representative, reason, status, _, terminal in representative_resolutions
        if status == "non_payload" and terminal is not None
    ]
    resolved_by_representative_chain = [
        {
            "chunk_uid": chunk_uid,
            "reason_code": reason,
            "representative_chain": chain,
            "terminal_chunk_uid": terminal,
        }
        for chunk_uid, _, reason, status, chain, terminal in representative_resolutions
        if status == "selected" and len(chain) > 1 and terminal is not None
    ]
    non_surviving_links = [
        {
            "chunk_uid": chunk_uid,
            "representative_chunk_uid": representative,
            "reason_code": reason,
            "resolution_status": status,
            "representative_chain": chain,
        }
        for chunk_uid, representative, reason, status, chain, _ in representative_resolutions
        if status not in {"selected", "non_payload"}
    ]
    removed_by_record: dict[str, list[JsonMap]] = {}
    for row in not_selected:
        removed_by_record.setdefault(str(row["stage_a_record_id"]), []).append(row)
    selected_by_record = {str(row["stage_a_record_id"]) for row in selected}
    zero_survivor_records: list[JsonMap] = []
    interaction_records: list[JsonMap] = []
    for record_id, rows in removed_by_record.items():
        reasons = sorted({str(row["stage_b_policy"].get("removed_reason")) for row in rows})
        has_curated_chunk = record_id in selected_by_record
        all_rows_explained = all(
            (
                is_authorized_terminal_non_selection(row)
                or (
                    isinstance(row["stage_b_policy"].get("representative_chunk_uid"), str)
                    and resolve_representative_chain(
                        str(row["stage_b_policy"]["representative_chunk_uid"])
                    )[0]
                    in {"selected", "non_payload"}
                )
            )
            for row in rows
        )
        if len(reasons) > 1:
            interaction_records.append(
                {
                    "stage_a_record_id": record_id,
                    "reason_codes": reasons,
                    "all_removals_explained": all_rows_explained,
                }
            )
        if not has_curated_chunk:
            zero_survivor_records.append(
                {
                    "stage_a_record_id": record_id,
                    "reason_codes": reasons,
                    "all_removals_explained": all_rows_explained,
                }
            )
    unexplained_zero_survivors = [
        item for item in zero_survivor_records if not item["all_removals_explained"]
    ]
    unexplained_interactions = [
        item for item in interaction_records if not item["all_removals_explained"]
    ]
    passed_ids = {str(row["chunk_uid"]) for row in passed}
    assert selected_ids.issubset(passed_ids)
    residual_payload_passed = all(
        len(str(row.get("text") or "").strip()) >= minimum_residual_chars
        for row in passed
        if row.get("stage_b_quality_candidate_transformations")
    )
    invariant_passed = not (
        missing_links
        or non_surviving_links
        or unexplained_zero_survivors
        or unexplained_interactions
        or not residual_payload_passed
    )
    return {
        "authority": "materialization_invariant",
        "selector_consumes_this_audit": False,
        "metadata_strata_or_target_mix_used": False,
        "representative_linkage": {
            "required_removed_chunks": len(required_links),
            "missing_representative_chunk_uids": missing_links,
            "representative_not_in_curated_pool": non_surviving_links,
            "resolved_by_non_payload_removal": resolved_by_non_payload_removal,
            "resolved_by_representative_chain": resolved_by_representative_chain,
            "passed": not missing_links and not non_surviving_links,
        },
        "residual_payload": {
            "span_rewrite_active": bool(span_transformations),
            "rewritten_chunks_checked": len({str(item["chunk_uid"]) for item in span_transformations}),
            "minimum_residual_chars": minimum_residual_chars,
            "invalid_residual_chunk_uids": [
                str(row["chunk_uid"])
                for row in passed
                if (
                    row.get("stage_b_quality_candidate_transformations")
                )
                and len(str(row.get("text") or "").strip()) < minimum_residual_chars
            ],
            "passed": residual_payload_passed,
            "zero_survivor_exception_allowed": False,
            "note": "Any promoted span rewrite must retain its chunk and preserve the declared Stage-B residual boundary.",
        },
        "zero_survivor_invariant": {
            "zero_survivor_records": zero_survivor_records,
            "unexplained_zero_survivor_records": unexplained_zero_survivors,
            "passed": not unexplained_zero_survivors,
        },
        "rule_interaction_audit": {
            "multi_reason_records": interaction_records,
            "unexplained_multi_reason_records": unexplained_interactions,
            "passed": not unexplained_interactions,
        },
        "passed": invariant_passed,
        "stage_c_delta_from_eligible_chunks": composition_audit[
            "delta_from_stage_b_pass"
        ]["stage_c_curated"],
        "claim_boundary": "Audits structural retention effects, representative linkage, and explainable zero-survivor outcomes; it never enforces or restores a target composition.",
    }


def materialize(config_path: Path, *, quality_scorer=None) -> JsonMap:
    root = Path(__file__).resolve().parent
    foundation = load_runtime_foundation(root)
    stage_tickets: list[RuntimeStageTicket] = []
    config = load_config(config_path)
    execution_scope = str(config.get("execution_scope") or "production")
    mode = resolve_curation_mode(
        str(config.get("curation_mode") or "normal"),
        execution_scope=execution_scope,
    )
    effective_policy = mode["effective_policy"]
    validate_run_policy_overrides(config, effective_policy)
    output_dir = Path(str(config["output_dir"]))
    quality_runtime = config.get("quality_runtime")
    quality_runtime = quality_runtime if isinstance(quality_runtime, dict) else {}
    quality_cache_path = Path(
        str(
            quality_runtime.get("observation_cache_path")
            or output_dir / "quality_teacher_observations.jsonl"
        )
    )
    redundancy_checkpoint_path = Path(
        str(
            quality_runtime.get("redundancy_checkpoint_path")
            or output_dir / "checkpoints" / "stage_b_redundancy_v2.json"
        )
    )
    preflight_runtime_writes(
        output_dir=output_dir,
        cache_path=quality_cache_path,
        checkpoint_path=redundancy_checkpoint_path,
    )
    source_specs = _source_specs(config["input"])
    source_paths = [_source_path(source) for source in source_specs]
    pretraining_audit = _pretraining_audit(config, source_paths)
    raw_records_by_source = [_read_jsonl([path]) for path in source_paths]
    raw_records = [record for records in raw_records_by_source for record in records]
    candidates = annotate_records(
        candidate
        for records, source in zip(raw_records_by_source, source_specs, strict=True)
        for candidate in adapt_raw_records(records, source)
    )
    stage_tickets.append(
        authorize_runtime_stage(
            foundation,
            RuntimeStageRequest(
                stage_id=StageId.STAGE_A,
                core_id=CoreId.VALIDITY,
                supplied_categories=(
                    "raw_text",
                    "declared_input_contract",
                    "deterministic_normalized_text",
                    "stable_identifiers",
                ),
            ),
        )
    )
    stage_a_settings = effective_policy["stage_a"]
    stage_a_policy = str(stage_a_settings["policy"])
    processed = annotate_records(
        process_candidate(candidate, index=index, stage_a_policy=stage_a_policy)
        for index, candidate in enumerate(candidates)
    )
    released = [row for row in processed if row["release_eligibility"]["eligible"]]
    quarantined = [row for row in processed if not row["release_eligibility"]["eligible"]]
    stage_b_policy = {
        **effective_policy["stage_b"],
        "max_chunk_chars": int(config["stage_b"]["max_chunk_chars"]),
    }
    stage_tickets.append(
        authorize_runtime_stage(
            foundation,
            RuntimeStageRequest(
                stage_id=StageId.STAGE_B,
                core_id=CoreId.REDUNDANCY,
                supplied_categories=(
                    "stage_a_survivors",
                    "deterministic_normalized_text",
                    "stable_identifiers",
                    "runtime_local_structural_evidence",
                    "prior_stage_reason_codes",
                ),
            ),
        )
    )
    passed, rejected = _stage_b_chunks(
        released,
        stage_b_policy,
        text_only=stage_a_policy == "text_only_v2",
    )
    stage_c_settings = config.get("stage_c") if isinstance(config.get("stage_c"), dict) else {}
    minimum_residual_chars = int(stage_c_settings["minimum_residual_chars"])
    span_transformations: list[JsonMap] = []
    quality_candidate_transformations: list[JsonMap] = []
    removal_policy = effective_policy["stage_b_policy"]
    stage_tickets.append(
        authorize_runtime_stage(
            foundation,
            RuntimeStageRequest(
                stage_id=StageId.STAGE_B,
                core_id=CoreId.QUALITY,
                supplied_categories=(
                    "stage_a_survivors",
                    "deterministic_normalized_text",
                    "stable_identifiers",
                    "runtime_local_structural_evidence",
                    "prior_stage_reason_codes",
                ),
            ),
        )
    )
    candidate_quality_rules = validate_quality_candidate_scope(removal_policy, execution_scope)
    quality_candidate_runtime_audit: JsonMap | None = None
    if "web_control_and_url_directory_span_candidate" in candidate_quality_rules:
        plan = build_web_span_plan(
            passed,
            minimum_residual_chars=minimum_residual_chars,
            token_counter=_token_proxy,
        )
        quality_candidate_runtime_audit = materialize_web_span_plan(
            passed,
            plan,
            token_counter=_token_proxy,
        )
        quality_candidate_transformations = quality_candidate_runtime_audit["transformations"]
        transformations_by_chunk: dict[str, list[JsonMap]] = {}
        for transformation in quality_candidate_transformations:
            transformations_by_chunk.setdefault(str(transformation["chunk_uid"]), []).append(transformation)
        passed = [dict(row) for row in quality_candidate_runtime_audit["records"]]
        for row in passed:
            traces = transformations_by_chunk.get(str(row["chunk_uid"]))
            if traces:
                row["stage_b_quality_candidate_transformations"] = traces
        span_transformations.extend(quality_candidate_transformations)
    selected, structural_not_selected, selection_audit = propose_stage_b_removals(
        passed, removal_policy
    )
    semantic_settings = stage_c_settings.get("semantic_coverage")
    semantic_provider = None
    restoration_candidate_uids: frozenset[str] | None = None
    semantic_artifact_preflight: JsonMap = {
        "status": "deterministic_lineage_guard_only"
    }
    if isinstance(semantic_settings, dict):
        parent_retained_path = semantic_settings.get("parent_retained_path")
        if isinstance(parent_retained_path, str) and parent_retained_path.strip():
            parent_rows = _read_jsonl((Path(parent_retained_path),))
            restoration_candidate_uids = frozenset(
                str(row["chunk_uid"]) for row in parent_rows
            )
        registry_path = Path(str(semantic_settings["provider_registry_path"]))
        registry = load_provider_registry(registry_path)
        provider_id = str(semantic_settings["provider_id"])
        semantic_provider = next(
            (item for item in registry.providers if item.provider_id == provider_id), None
        )
        if semantic_provider is None:
            raise RuntimeError(f"Unknown semantic Coverage provider: {provider_id}")
        structural_universe = _stage_b_materialization_universe(
            passed, selected, structural_not_selected
        )
        semantic_artifact_preflight = validate_semantic_coverage_artifacts(
            universe=structural_universe,
            corpus_path=Path(str(semantic_settings["corpus_path"])),
            graph_path=Path(str(semantic_settings["graph_path"])),
            provider=semantic_provider,
        )
    redundancy_settings_payload = dict(effective_policy["redundancy_v2"]["settings"])
    checkpointed_redundancy = load_or_build_redundancy(
        selected,
        mode=RedundancyMode(mode["mode"]),
        settings=RedundancySettings(**redundancy_settings_payload),
        checkpoint_path=redundancy_checkpoint_path,
    )
    redundancy_result = checkpointed_redundancy.result
    panel_path = Path(str(effective_policy["quality_runtime"]["oracle_panel"]))
    if not panel_path.is_absolute():
        panel_path = root / panel_path
    dotenv_path = Path(str(quality_runtime.get("dotenv_path") or root.parent / ".env"))
    if quality_scorer is not None:
        quality_results, quality_scoring_audit = quality_scorer(
            list(redundancy_result.survivors),
            panel_path=panel_path,
            dotenv_path=dotenv_path,
            cache_path=quality_cache_path,
            task_workers=int(quality_runtime.get("task_workers") or 8),
        )
    else:
        method = str(quality_runtime.get("method") or "")
        if method != "distilled_quality_ranker_v1":
            raise RuntimeError(
                "Curation runtime requires quality_runtime.method=distilled_quality_ranker_v1; "
                "the Teacher panel is calibration/oracle-only"
            )
        embedding_manifest_path = Path(str(quality_runtime["embedding_manifest_path"]))
        ranker_manifest_path = Path(str(quality_runtime["ranker_manifest_path"]))
        if not embedding_manifest_path.is_absolute():
            embedding_manifest_path = root / embedding_manifest_path
        if not ranker_manifest_path.is_absolute():
            ranker_manifest_path = root / ranker_manifest_path
        quality_results, quality_scoring_audit = score_quality_runtime(
            QualityRuntimeRequest(
                rows=tuple(redundancy_result.survivors),
                distilled=DistilledRuntimeConfig(
                    embedding_manifest_path=embedding_manifest_path,
                    ranker_manifest_path=ranker_manifest_path,
                    oracle_fallback_enabled=bool(
                        quality_runtime.get("oracle_fallback_enabled", True)
                    ),
                    maximum_oracle_fraction=float(
                        quality_runtime.get("maximum_oracle_fraction", 0.05)
                    ),
                ),
                teacher=TeacherOracleConfig(
                    panel_path=panel_path,
                    dotenv_path=dotenv_path,
                    cache_path=quality_cache_path,
                    task_workers=int(quality_runtime.get("task_workers") or 8),
                ),
            )
        )
    quality_result = apply_quality_policy(
        redundancy_result.survivors,
        results_by_chunk=quality_results,
        mode=CurationMode(mode["mode"]),
    )
    selected = [dict(row) for row in quality_result.survivors]
    not_selected = [
        *structural_not_selected,
        *redundancy_result.removals,
        *quality_result.not_selected,
    ]
    stage_b_proposed_survivors = [dict(row) for row in selected]
    stage_b_materialization_universe = _stage_b_materialization_universe(
        passed, selected, not_selected
    )
    stage_tickets.append(
        authorize_runtime_stage(
            foundation,
            RuntimeStageRequest(
                stage_id=StageId.STAGE_C,
                core_id=CoreId.COVERAGE,
                supplied_categories=(
                    "stage_b_survivors",
                    "typed_non_selection_proposals",
                    "representative_links",
                    "prior_stage_reason_codes",
                ),
            ),
        )
    )
    semantic_coverage_audit: JsonMap = {
        "status": "deterministic_lineage_guard_only",
        "semantic_graph_consumed": False,
        "scientific_promotion_claimed": False,
        "artifact_preflight": semantic_artifact_preflight,
    }
    if isinstance(semantic_settings, dict):
        if semantic_provider is None:
            raise RuntimeError("Semantic Coverage provider preflight was not completed")
        selected, semantic_coverage_audit = materialize_semantic_coverage(
            universe=stage_b_materialization_universe,
            proposed_survivors=tuple(selected),
            non_selection_proposals=tuple(not_selected),
            corpus_path=Path(str(semantic_settings["corpus_path"])),
            graph_path=Path(str(semantic_settings["graph_path"])),
            provider=semantic_provider,
            execution_scope=CoverageExecutionScope(execution_scope),
            restoration_candidate_uids=restoration_candidate_uids,
            representative_families=coverage_families_from_redundancy_plan(
                redundancy_result.plan
            ),
        )
        semantic_coverage_audit["status"] = "semantic_coverage_materialized"
        semantic_coverage_audit["semantic_graph_consumed"] = True
        semantic_coverage_audit["scientific_promotion_claimed"] = False
        semantic_coverage_audit["artifact_preflight"] = semantic_artifact_preflight
    selected_ids = {str(row["chunk_uid"]) for row in selected}
    final_not_selected = [
        row for row in not_selected if str(row["chunk_uid"]) not in selected_ids
    ]
    paths = {
        "stage_a_release": output_dir / "stage_a_release_candidates.jsonl",
        "stage_a_quarantine": output_dir / "stage_a_quarantined_candidates.jsonl",
        "stage_b_pass": output_dir / "stage_b_pass_chunks.jsonl",
        "stage_b_rejected": output_dir / "stage_b_rejected_chunks.jsonl",
        "stage_b_proposed_survivors": output_dir / "stage_b_proposed_survivors.jsonl",
        "stage_b_non_selection_proposals": output_dir / "stage_b_non_selection_proposals.jsonl",
        "stage_c_not_selected": output_dir / "stage_c_not_selected_chunks.jsonl",
        "stage_c_curated": output_dir / "stage_c_curated_chunks.jsonl",
        "stage_b_quality_candidate_transformations": output_dir / "stage_b_quality_candidate_transformations.jsonl",
    }
    _write_jsonl(paths["stage_a_release"], released)
    _write_jsonl(paths["stage_a_quarantine"], quarantined)
    _write_jsonl(paths["stage_b_pass"], passed)
    _write_jsonl(paths["stage_b_rejected"], rejected)
    _write_jsonl(paths["stage_b_proposed_survivors"], stage_b_proposed_survivors)
    _write_jsonl(paths["stage_b_non_selection_proposals"], not_selected)
    _write_jsonl(paths["stage_c_not_selected"], final_not_selected)
    _write_jsonl(paths["stage_c_curated"], selected)
    _write_jsonl(paths["stage_b_quality_candidate_transformations"], quality_candidate_transformations)
    semantic_coverage_audit_path = output_dir / "stage_c_coverage_audit.json"
    save_json(semantic_coverage_audit_path, semantic_coverage_audit)
    paths["stage_c_coverage_audit"] = semantic_coverage_audit_path
    composition_audit = build_composition_audit(
        {
            "raw_input": candidates,
            "stage_a_release": released,
            "stage_b_pass": passed,
            "stage_c_curated": selected,
        }
    )
    explanatory_composition = build_composition_artifacts(
        tuple(
            CompositionRecord(
                str(row.get("chunk_uid") or f"eligible-{index}"),
                str(row.get("text") or ""),
                int(row.get("token_proxy") or len(str(row.get("text") or "").split())),
            )
            for index, row in enumerate(passed)
        ),
        tuple(
            CompositionRecord(
                str(row.get("chunk_uid") or f"curated-{index}"),
                str(row.get("text") or ""),
                int(row.get("token_proxy") or len(str(row.get("text") or "").split())),
            )
            for index, row in enumerate(selected)
        ),
    )
    composition_paths = write_composition_artifacts(
        explanatory_composition, output_dir
    )
    paths.update(
        {
            "composition_audit_json": composition_paths.audit_json,
            "composition_by_route_csv": composition_paths.route_csv,
            "composition_by_language_csv": composition_paths.language_csv,
            "eligible_curated_composition_delta_csv": composition_paths.delta_csv,
        }
    )
    stage_a_role = (
        "source_agnostic_text_normalization_and_integrity_handling"
        if stage_a_policy == "text_only_v2"
        else "candidate_provenance_normalization_and_risk_quarantine"
    )
    coverage_impact_audit = _coverage_impact_audit(
        passed=passed,
        selected=selected,
        rejected=rejected,
        not_selected=final_not_selected,
        span_transformations=span_transformations,
        minimum_residual_chars=minimum_residual_chars,
        composition_audit=composition_audit,
    )
    if not coverage_impact_audit["passed"]:
        raise RuntimeError("Coverage invariant failed: materialization has an unexplained removal outcome.")
    report = {
        "schema_version": "curation-materialization-report-v1",
        "status": "curation_materialization_complete",
        "stage_contract": {
            "stage_a": stage_a_role,
            "stage_b": "redundancy_removal_and_positive_quality_selection",
            "stage_c": "coverage_veto_and_final_materialization",
            "external_evaluation": "not_started",
        },
        "curation_mode": mode,
        "framework_runtime": build_foundation_report(
            foundation,
            tuple(stage_tickets),
        ).model_dump(mode="json"),
        "effective_policy_manifest": {
            "profile_id": mode["profile_id"],
            "profile_status": mode["profile_status"],
            "enabled_policy_ids": mode["enabled_policy_ids"],
            "policy_sha256": mode["effective_policy_sha256"],
            "policy": effective_policy,
        },
        "development_candidate_profile": {
            "id": config.get("development_candidate_profile"),
            "enabled_quality_policy_keys": candidate_quality_rules,
            "runtime_active": False,
        } if candidate_quality_rules else None,
        "summary": {
            "input_records": len(raw_records),
            "stage_a_release_records": len(released),
            "stage_a_quarantined_records": len(quarantined),
            "stage_b_hard_gate_pass_chunks": len(passed),
            "stage_b_rejected_chunks": len(rejected),
            "stage_c_retained_chunks": len(selected),
            "stage_c_not_selected_chunks": len(final_not_selected),
            "stage_b_near_duplicate_removed_chunks": int(
                redundancy_result.audit["removal_witness_counts"].get(
                    "bounded_near_substitute", 0
                )
            ) + int(
                redundancy_result.audit["removal_witness_counts"].get(
                    "token_preserving_prose_reflow", 0
                )
            ),
            "stage_b_redundancy_v2_removed_chunks": len(redundancy_result.removals),
            "stage_b_quality_not_selected_chunks": len(quality_result.not_selected),
            "stage_b_structural_scaffold_removed_chunks": int(selection_audit["structural_scaffold_removed_chunks"]),
            "stage_b_explicit_generated_artifact_removed_chunks": int(selection_audit["explicit_generated_artifact_removed_chunks"]),
            "stage_b_license_comment_only_removed_chunks": int(selection_audit["license_comment_only_removed_chunks"]),
            "stage_b_empty_html_shell_removed_chunks": int(selection_audit["empty_html_shell_removed_chunks"]),
            "stage_b_web_chrome_only_removed_chunks": int(selection_audit["web_chrome_only_removed_chunks"]),
            "stage_b_explicit_non_payload_rejected_chunks": int(
                selection_audit["quality_retention"]["decision_counts"]["reject"]
            ),
            "stage_b_positive_quality_kept_chunks": int(
                quality_result.audit["chunks_meeting_retention_threshold"]
            ),
            "stage_b_all_quality_policies_passed_chunks": int(
                quality_result.audit["chunks_with_all_policy_pass"]
            ),
            "stage_b_quality_chunks_with_any_abstain": int(
                quality_result.audit["chunks_with_any_abstain"]
            ),
            "stage_b_quality_candidate_span_transformations": len(quality_candidate_transformations),
            "stage_b_total_span_transformations": len(span_transformations),
            "stage_c_curated_chunks": len(selected),
            "stage_c_curated_whitespace_token_proxy": sum(int(row["token_proxy"]) for row in selected),
            "stage_a_quarantine_reasons": dict(Counter(reason for row in quarantined for reason in row["quarantine"]["reasons"])),
            "stage_b_hard_gate_rejection_reasons": dict(Counter(reason for row in rejected for reason in row["stage_b_hard_gate_reasons"])),
        },
        "composition_audit": composition_audit,
        "composition_artifacts": {
            "authority": explanatory_composition.authority,
            "consumed_by_selection": explanatory_composition.consumed_by_selection,
            "target_distribution_enforced": explanatory_composition.target_distribution_enforced,
            "baseline_stage": explanatory_composition.baseline_stage,
            "comparison_unit": explanatory_composition.comparison_unit,
            "eligible_tokens": explanatory_composition.eligible_tokens,
            "curated_tokens": explanatory_composition.curated_tokens,
        },
        "reason_code_impact_audit": build_reason_code_impact_audit(
            quarantined, rejected, final_not_selected, span_transformations
        ),
        "coverage_impact_audit": coverage_impact_audit,
        "measurement_contract": {
            "runtime_token_measurement": "whitespace_proxy_non_training",
            "exact_tokenizer_count": None,
            "exact_tokenizer_count_role": "external_evaluation_only_with_declared_tokenizer",
        },
        "selector_boundary": {
            "utility_read": False,
            "benchmark_outcomes_read": False,
            "source_pool_role_read": False,
            "composition_read": False,
        },
        "policy_fingerprint": _policy_fingerprint(),
        "stage_b_policy": selection_audit,
        "stage_b_redundancy_v2": redundancy_result.audit,
        "stage_b_redundancy_checkpoint": {
            "checkpoint_hit": checkpointed_redundancy.checkpoint_hit,
            "identity_sha256": checkpointed_redundancy.identity_sha256,
            "checkpoint_path": checkpointed_redundancy.checkpoint_path,
        },
        "stage_b_quality": quality_result.audit,
        "quality_scoring": quality_scoring_audit,
        "stage_c_coverage": semantic_coverage_audit,
        "quality_candidate_runtime_audit": quality_candidate_runtime_audit,
        "pretraining_audit": pretraining_audit,
        "claim_boundary": config["claim_boundary"],
        "outputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "source_sha256": {
            "config": sha256_file(config_path),
            "inputs": {str(path): sha256_file(path) for path in source_paths},
        },
    }
    save_json(output_dir / "curation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize a frozen Stage A/B/C curation output.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(args.config)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
