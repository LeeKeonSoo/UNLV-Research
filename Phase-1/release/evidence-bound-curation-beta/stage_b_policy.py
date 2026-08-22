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
STRUCTURAL_SCAFFOLD_REASON = "structural_scaffold_representative_retained"
STAGE_B_STRUCTURAL_POLICY_REASON_CODES = {
    "stage_b_explicit_generated_artifact": frozenset(
        {EXPLICIT_GENERATED_ARTIFACT_REASON}
    ),
    "stage_b_license_comment_only": frozenset({LICENSE_COMMENT_ONLY_REASON}),
    "stage_b_structural_scaffold": frozenset({STRUCTURAL_SCAFFOLD_REASON}),
    "stage_b_empty_html_shell": frozenset({EMPTY_HTML_SHELL_REASON}),
    "stage_b_web_chrome_only_chunk": frozenset({WEB_CHROME_ONLY_REASON}),
}
IMPORT_RE = re.compile(
    r"^(?:from\s+\S+\s+import\s+|import\s+|require\s*\(|include\s+|using\s+)"
)
METADATA_RE = re.compile(r"^(?:__all__|__version__|__path__|export\s+)")


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


def _normalize_stage_b_row(source: JsonMap) -> JsonMap:
    row = dict(source)
    metadata = row.pop("stage_c_policy_metadata", None)
    if isinstance(metadata, dict):
        row["stage_b_policy_metadata"] = metadata
    visible = row.pop("stage_c_selector_visible", None)
    if isinstance(visible, dict):
        row["stage_b_selector_visible"] = visible
    row["stage_b_policy"] = _selection_trace(
        accepted=True,
        trigger="no_structural_nonpayload_evidence",
        non_trigger_boundary="no_active_structural_policy_triggered",
    )
    return row


def _scaffold_signature(text: str) -> str | None:
    code_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "//", "/*", "*"))
    ]
    import_count = sum(bool(IMPORT_RE.match(line)) for line in code_lines)
    metadata_count = sum(bool(METADATA_RE.match(line)) for line in code_lines)
    if len(code_lines) < 4 or import_count < 2 or import_count + metadata_count != len(
        code_lines
    ):
        return None
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _compact_scaffold_families(
    selected: list[JsonMap], removed: list[JsonMap]
) -> tuple[list[JsonMap], int]:
    families: dict[str, list[JsonMap]] = defaultdict(list)
    for row in selected:
        signature = _scaffold_signature(str(row["text"]))
        if signature is not None:
            families[signature].append(row)
    removed_ids: set[str] = set()
    for signature, family in families.items():
        ordered = sorted(family, key=lambda row: str(row["chunk_uid"]))
        if len(ordered) < 2:
            continue
        representative = ordered[0]
        representative["stage_b_policy"].update(
            {
                "accepted_by": "structural_scaffold_family_representative",
                "trigger": "identical_normalized_structural_scaffold_family",
                "non_trigger_boundary": "stable_family_representative_retained",
                "structural_scaffold_signature": signature,
            }
        )
        for row in ordered[1:]:
            removed_ids.add(str(row["chunk_uid"]))
            row["stage_b_policy"] = _selection_trace(
                accepted=False,
                trigger="identical_normalized_structural_scaffold_family",
                non_trigger_boundary="distinct_scaffold_signatures_are_retained",
                reason_code=STRUCTURAL_SCAFFOLD_REASON,
                token_delta_proxy=-int(
                    row.get("token_proxy") or len(str(row["text"]).split())
                ),
                representative_chunk_uid=str(representative["chunk_uid"]),
            )
            row["stage_b_policy"]["structural_scaffold_signature"] = signature
            removed.append(row)
    return [row for row in selected if str(row["chunk_uid"]) not in removed_ids], len(
        removed_ids
    )


def _apply_explicit_nonpayload_rules(
    selected: list[JsonMap], removed: list[JsonMap], settings: JsonMap
) -> tuple[list[JsonMap], JsonMap]:
    decisions, audit = evaluate_quality_retention(selected, settings)
    retained: list[JsonMap] = []
    for row in selected:
        decision = decisions[str(row["chunk_uid"])]
        row["quality_retention_decision"] = decision
        if decision["decision"] != QUALITY_REJECT:
            row["stage_b_policy"]["quality_retention_decision"] = decision["decision"]
            retained.append(row)
            continue
        row["stage_b_policy"] = _selection_trace(
            accepted=False,
            trigger=str(decision["trigger"]),
            non_trigger_boundary=str(decision["non_trigger_boundary"]),
            reason_code=str(decision["reason_code"]),
            token_delta_proxy=-int(
                row.get("token_proxy") or len(str(row["text"]).split())
            ),
        )
        row["stage_b_policy"]["quality_policy_id"] = decision["policy_id"]
        row["stage_b_policy"]["artifact_evidence"] = decision["evidence"]
        removed.append(row)
    return retained, audit


def propose_stage_b_removals(
    chunks: Iterable[JsonMap], config: JsonMap
) -> tuple[list[JsonMap], list[JsonMap], JsonMap]:
    scaffold_settings = dict(config.get("structural_scaffold_compaction") or {})
    artifact_settings = dict(config.get("structural_artifact_rules") or {})
    selected = [
        _normalize_stage_b_row(row)
        for row in sorted(chunks, key=lambda item: str(item["chunk_uid"]))
    ]
    removed: list[JsonMap] = []
    scaffold_enabled = bool(scaffold_settings.get("enabled", False))
    scaffold_removed = 0
    if scaffold_enabled:
        selected, scaffold_removed = _compact_scaffold_families(selected, removed)
    selected, quality_audit = _apply_explicit_nonpayload_rules(
        selected, removed, artifact_settings
    )
    reason_counts = quality_audit["reason_code_counts"]
    return selected, removed, {
        "selection_mode": "stage_b_reason_coded_structural_without_budget",
        "owner_stage": "Stage B",
        "core_ids": ["redundancy", "quality"],
        "proposal_is_final_membership": False,
        "structural_scaffold_compaction": {"enabled": scaffold_enabled},
        "structural_scaffold_removed_chunks": scaffold_removed,
        "explicit_generated_artifact_removed_chunks": reason_counts.get(
            EXPLICIT_GENERATED_ARTIFACT_REASON, 0
        ),
        "license_comment_only_removed_chunks": reason_counts.get(
            LICENSE_COMMENT_ONLY_REASON, 0
        ),
        "empty_html_shell_removed_chunks": reason_counts.get(
            EMPTY_HTML_SHELL_REASON, 0
        ),
        "web_chrome_only_removed_chunks": reason_counts.get(WEB_CHROME_ONLY_REASON, 0),
        "quality_retention": quality_audit,
        "structural_artifact_rules": artifact_settings,
        "coverage_representatives_retained": scaffold_removed,
        "budget_applied": False,
        "utility_read": False,
        "benchmark_outcomes_read": False,
        "source_pool_role_read": False,
    }


__all__ = ["STAGE_B_STRUCTURAL_POLICY_REASON_CODES", "propose_stage_b_removals"]
