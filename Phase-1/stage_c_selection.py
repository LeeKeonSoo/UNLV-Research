from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from quality_retention import QUALITY_REJECT, evaluate_quality_retention
from quality_rule_evidence import (
    EMPTY_HTML_SHELL_REASON,
    EXPLICIT_GENERATED_ARTIFACT_REASON,
    LICENSE_COMMENT_ONLY_REASON,
    WEB_CHROME_ONLY_REASON,
)


JsonMap = dict[str, Any]
STAGE_C_NEAR_DUPLICATE_REASON = "near_duplicate_representative_retained"
STAGE_C_GENERATED_ARTIFACT_REASON = EXPLICIT_GENERATED_ARTIFACT_REASON
STAGE_C_LICENSE_COMMENT_ONLY_REASON = LICENSE_COMMENT_ONLY_REASON
STAGE_C_STRUCTURAL_SCAFFOLD_REASON = "structural_scaffold_representative_retained"
STAGE_C_EMPTY_HTML_SHELL_REASON = EMPTY_HTML_SHELL_REASON
STAGE_C_WEB_CHROME_ONLY_REASON = WEB_CHROME_ONLY_REASON
STAGE_C_POLICY_REASON_CODES = {
    "stage_c_symmetric_near_duplicate": frozenset({STAGE_C_NEAR_DUPLICATE_REASON}),
    "stage_c_explicit_generated_artifact": frozenset({STAGE_C_GENERATED_ARTIFACT_REASON}),
    "stage_c_license_comment_only": frozenset({STAGE_C_LICENSE_COMMENT_ONLY_REASON}),
    "stage_c_structural_scaffold": frozenset({STAGE_C_STRUCTURAL_SCAFFOLD_REASON}),
    "stage_c_empty_html_shell": frozenset({STAGE_C_EMPTY_HTML_SHELL_REASON}),
    "stage_c_web_chrome_only_chunk": frozenset({STAGE_C_WEB_CHROME_ONLY_REASON}),
}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IMPORT_RE = re.compile(r"^(?:from\s+\S+\s+import\s+|import\s+|require\s*\(|include\s+|using\s+)")
METADATA_RE = re.compile(r"^(?:__all__|__version__|__path__|export\s+)")


def _token_shingles(text: str, size: int) -> frozenset[bytes]:
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    return frozenset(
        hashlib.blake2b(" ".join(tokens[index : index + size]).encode("utf-8"), digest_size=8).digest()
        for index in range(max(0, len(tokens) - size + 1))
    )


def _index_keys(shingles: frozenset[bytes]) -> set[bytes]:
    ordered = sorted(shingles)
    if not ordered:
        return set()
    positions = {0, len(ordered) // 4, len(ordered) // 2, (3 * len(ordered)) // 4, len(ordered) - 1}
    return {ordered[position] for position in positions}


def _is_near_duplicate(left: frozenset[bytes], right: frozenset[bytes], threshold: float) -> bool:
    if not left or not right:
        return False
    overlap = len(left & right)
    return (overlap / len(left)) >= threshold and (overlap / len(right)) >= threshold


def _is_structural_scaffold(text: str) -> bool:
    code_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//", "/*", "*"))
    ]
    import_count = sum(bool(IMPORT_RE.match(line)) for line in code_lines)
    metadata_count = sum(bool(METADATA_RE.match(line)) for line in code_lines)
    return len(code_lines) >= 4 and import_count >= 2 and import_count + metadata_count == len(code_lines)


def _structural_scaffold_signature(text: str) -> str | None:
    """Return an exact normalized scaffold-family signature, never a broad category label."""
    if not _is_structural_scaffold(text):
        return None
    code_lines = [
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//", "/*", "*"))
    ]
    normalized = "\n".join(code_lines).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _selection_trace(
    *,
    accepted: bool,
    trigger: str,
    non_trigger_boundary: str,
    reason_code: str | None = None,
    token_delta_proxy: int = 0,
    representative_chunk_uid: str | None = None,
) -> JsonMap:
    trace: JsonMap = {
        "accepted": accepted,
        "trigger": trigger,
        "non_trigger_boundary": non_trigger_boundary,
        "token_delta_proxy": token_delta_proxy,
        "budget_applied": False,
        "utility_read": False,
        "benchmark_outcomes_read": False,
    }
    if reason_code is not None:
        trace["accepted_by"] = reason_code
        trace["removed_reason"] = reason_code
    else:
        trace["accepted_by"] = trigger
    if representative_chunk_uid is not None:
        trace["representative_chunk_uid"] = representative_chunk_uid
    return trace


def _compact_structural_scaffolds(selected: list[JsonMap], removed: list[JsonMap]) -> tuple[list[JsonMap], int]:
    families: dict[str, list[JsonMap]] = defaultdict(list)
    for row in selected:
        signature = _structural_scaffold_signature(str(row["text"]))
        if signature is not None:
            families[signature].append(row)
    removed_ids: set[str] = set()
    for signature, family in families.items():
        ordered = sorted(family, key=lambda row: str(row["chunk_uid"]))
        if len(ordered) < 2:
            continue
        representative = ordered[0]
        representative["stage_c_selection"].update(
            {
                "accepted_by": "structural_scaffold_family_representative",
                "trigger": "identical_normalized_structural_scaffold_family",
                "non_trigger_boundary": "stable_family_representative_retained",
                "structural_scaffold_signature": signature,
            }
        )
        for row in ordered[1:]:
            chunk_uid = str(row["chunk_uid"])
            removed_ids.add(chunk_uid)
            row["stage_c_selection"] = _selection_trace(
                accepted=False,
                trigger="identical_normalized_structural_scaffold_family",
                non_trigger_boundary="distinct_scaffold_signatures_are_retained",
                reason_code=STAGE_C_STRUCTURAL_SCAFFOLD_REASON,
                token_delta_proxy=-int(row.get("token_proxy") or len(str(row["text"]).split())),
                representative_chunk_uid=str(representative["chunk_uid"]),
            )
            row["stage_c_selection"]["structural_scaffold_signature"] = signature
            removed.append(row)
    return [row for row in selected if str(row["chunk_uid"]) not in removed_ids], len(removed_ids)


def _apply_quality_retention(
    selected: list[JsonMap],
    removed: list[JsonMap],
    artifact_settings: JsonMap,
) -> tuple[list[JsonMap], JsonMap]:
    decisions, audit = evaluate_quality_retention(selected, artifact_settings)
    retained: list[JsonMap] = []
    for row in selected:
        decision = decisions[str(row["chunk_uid"])]
        row["quality_retention_decision"] = decision
        if decision["decision"] == QUALITY_REJECT:
            row["stage_c_selection"] = _selection_trace(
                accepted=False,
                trigger=str(decision["trigger"]),
                non_trigger_boundary=str(decision["non_trigger_boundary"]),
                reason_code=str(decision["reason_code"]),
                token_delta_proxy=-int(row.get("token_proxy") or len(str(row["text"]).split())),
            )
            row["stage_c_selection"]["quality_policy_id"] = decision["policy_id"]
            row["stage_c_selection"]["artifact_evidence"] = decision["evidence"]
            removed.append(row)
            continue
        row["stage_c_selection"]["quality_retention_decision"] = decision["decision"]
        retained.append(row)
    return retained, audit


def select_chunks(chunks: Iterable[JsonMap], config: JsonMap) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    selection_settings = dict(config)
    if "operational_selection" in selection_settings:
        raise RuntimeError(
            "Weighted operational-priority selection was retired. "
            "Stage C supports only declared reason-coded duplicate and scaffold rules."
        )
    duplicate_settings = dict(selection_settings.get("near_duplicate_compaction") or {})
    artifact_settings = dict(selection_settings.get("structural_artifact_rules") or {})
    if "coverage_guard" in selection_settings:
        raise RuntimeError(
            "Coverage is audit-only and cannot select or retain metadata strata. "
            "Use the materialization coverage audit to inspect representative linkage."
        )
    generated_artifact_enabled = bool(artifact_settings.get("explicit_generated_artifact", False))
    license_comment_only_enabled = bool(artifact_settings.get("license_comment_only_chunk", False))
    empty_html_shell_enabled = bool(artifact_settings.get("empty_html_shell", False))
    web_chrome_only_enabled = bool(artifact_settings.get("web_chrome_only_chunk", False))
    shingle_size = int(duplicate_settings.get("shingle_size", 5))
    min_tokens = int(duplicate_settings.get("minimum_lexical_tokens", 40))
    threshold = float(duplicate_settings.get("symmetric_overlap_threshold", 0.95))
    near_duplicate_enabled = bool(duplicate_settings.get("candidate_enabled", False))
    if shingle_size < 2 or min_tokens < shingle_size or not 0.0 < threshold <= 1.0:
        raise RuntimeError("Invalid near-duplicate compaction contract")
    selected: list[JsonMap] = []
    removed: list[JsonMap] = []
    signatures: list[frozenset[bytes]] = []
    representative_ids: list[str] = []
    buckets: dict[bytes, list[int]] = defaultdict(list)
    for raw in sorted(chunks, key=lambda row: str(row["chunk_uid"])):
        row = dict(raw)
        tokens = TOKEN_RE.findall(str(row["text"]))
        signature = _token_shingles(str(row["text"]), shingle_size) if len(tokens) >= min_tokens else frozenset()
        candidates = {index for key in _index_keys(signature) for index in buckets[key]}
        representative_index = (
            next(
                (index for index in sorted(candidates) if _is_near_duplicate(signature, signatures[index], threshold)),
                None,
            )
            if near_duplicate_enabled
            else None
        )
        if representative_index is None:
            row["stage_c_selection"] = _selection_trace(
                accepted=True,
                trigger="no_symmetric_near_duplicate_match",
                non_trigger_boundary="symmetric_overlap_below_declared_threshold",
            )
            selected.append(row)
            signatures.append(signature)
            representative_ids.append(str(row["chunk_uid"]))
            for key in _index_keys(signature):
                buckets[key].append(len(signatures) - 1)
            continue
        row["stage_c_selection"] = _selection_trace(
            accepted=False,
            trigger="symmetric_shingle_containment_at_or_above_declared_threshold",
            non_trigger_boundary="no_coverage_representative_required",
            reason_code=STAGE_C_NEAR_DUPLICATE_REASON,
            token_delta_proxy=-int(row.get("token_proxy") or len(str(row["text"]).split())),
            representative_chunk_uid=representative_ids[representative_index],
        )
        row["stage_c_selection"]["duplicate_representative_chunk_uid"] = representative_ids[representative_index]
        removed.append(row)
    selected, coverage_representatives = _compact_structural_scaffolds(selected, removed)
    selected, quality_retention_audit = _apply_quality_retention(selected, removed, artifact_settings)
    quality_reason_counts = quality_retention_audit["reason_code_counts"]
    near_duplicate_removed = sum(
        row["stage_c_selection"].get("removed_reason") == STAGE_C_NEAR_DUPLICATE_REASON
        for row in removed
    )
    scaffold_removed = sum(
        row["stage_c_selection"].get("removed_reason") == STAGE_C_STRUCTURAL_SCAFFOLD_REASON
        for row in removed
    )
    return selected, removed, {
        "selection_mode": (
            "reason_coded_duplicate_generated_artifact_and_scaffold_compaction_without_budget"
            if generated_artifact_enabled
            or license_comment_only_enabled
            or empty_html_shell_enabled
            or web_chrome_only_enabled
            else "reason_coded_duplicate_and_scaffold_compaction_without_budget"
        ),
        "near_duplicate_compaction": {
            "candidate_enabled": near_duplicate_enabled,
            "shingle_size": shingle_size,
            "minimum_lexical_tokens": min_tokens,
            "symmetric_overlap_threshold": threshold,
        },
        "near_duplicate_removed_chunks": near_duplicate_removed,
        "structural_scaffold_removed_chunks": scaffold_removed,
        "explicit_generated_artifact_removed_chunks": quality_reason_counts.get(STAGE_C_GENERATED_ARTIFACT_REASON, 0),
        "license_comment_only_removed_chunks": quality_reason_counts.get(STAGE_C_LICENSE_COMMENT_ONLY_REASON, 0),
        "empty_html_shell_removed_chunks": quality_reason_counts.get(STAGE_C_EMPTY_HTML_SHELL_REASON, 0),
        "web_chrome_only_removed_chunks": quality_reason_counts.get(STAGE_C_WEB_CHROME_ONLY_REASON, 0),
        "quality_retention": quality_retention_audit,
        "structural_artifact_rules": {
            "explicit_generated_artifact": generated_artifact_enabled,
            "license_comment_only_chunk": license_comment_only_enabled,
            "empty_html_shell": empty_html_shell_enabled,
            "web_chrome_only_chunk": web_chrome_only_enabled,
        },
        "coverage_representatives_retained": coverage_representatives,
        "coverage_invariants": {
            "authority": "audit_only",
            "selector_consumes_invariants": False,
            "metadata_strata_supported": False,
            "representative_links_are_reported_by": "materialization_coverage_impact_audit",
        },
        "budget_applied": False,
        "utility_read": False,
        "benchmark_outcomes_read": False,
        "source_pool_role_read": False,
    }
