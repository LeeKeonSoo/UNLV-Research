#!/usr/bin/env python3
"""Audit whether a corpus exposes evidence for additional selection rules.

This is deliberately diagnostic-only.  A candidate count or a path pattern does
not authorize a new removal policy; the report makes missing provenance labels
and known false-positive risks explicit before a rule can be proposed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from build_core_rule_inventory import PATH_RULES, _evidence


JsonMap = dict[str, Any]


def _read_jsonl(path: Path) -> Iterable[JsonMap]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _token_proxy(row: JsonMap) -> int:
    value = row.get("token_count")
    if isinstance(value, int) and value >= 0:
        return value
    return max(1, len(str(row.get("text") or "").split()))


def build_report(rows: Iterable[JsonMap]) -> JsonMap:
    total_chunks = 0
    total_tokens = 0
    metadata_present: Counter[str] = Counter()
    metadata_values: dict[str, Counter[str]] = {
        "artifact_context.generation": Counter(),
        "artifact_context.dependency_copy": Counter(),
        "stage_c_policy_metadata.content_type": Counter(),
        "audit_only_metadata.provenance.license": Counter(),
        "audit_only_metadata.provenance.source_name": Counter(),
    }
    candidate_counts: Counter[str] = Counter()
    candidate_tokens: Counter[str] = Counter()

    for row in rows:
        total_chunks += 1
        token_proxy = _token_proxy(row)
        total_tokens += token_proxy
        partition = row.get("partition") if isinstance(row.get("partition"), dict) else {}
        policy_metadata = row.get("stage_c_policy_metadata")
        if not isinstance(policy_metadata, dict):
            policy_metadata = {}
        artifact_context = row.get("artifact_context")
        if not isinstance(artifact_context, dict):
            artifact_context = {}
        audit_only_metadata = row.get("audit_only_metadata")
        if not isinstance(audit_only_metadata, dict):
            audit_only_metadata = {}
        provenance = audit_only_metadata.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
        metadata = {
            "path": policy_metadata.get("path") or partition.get("path"),
            "artifact_context.generation": policy_metadata.get("declared_generation") if "declared_generation" in policy_metadata else artifact_context.get("generation"),
            "artifact_context.dependency_copy": policy_metadata.get("declared_dependency_copy") if "declared_dependency_copy" in policy_metadata else artifact_context.get("dependency_copy"),
            "stage_c_policy_metadata.content_type": policy_metadata.get("content_type") or partition.get("content_type"),
            "audit_only_metadata.provenance.license": provenance.get("license"),
            "audit_only_metadata.provenance.source_name": provenance.get("source_name"),
        }
        for key, value in metadata.items():
            if value is not None and str(value) != "":
                metadata_present[key] += 1
                if key in metadata_values:
                    metadata_values[key][str(value).lower()] += 1

        if not policy_metadata:
            policy_metadata = {"path": partition.get("path"), "content_type": partition.get("content_type")}
        evidence = _evidence(str(row.get("text") or ""), policy_metadata)
        for key in (
            "strong_generated_marker_candidate",
            "one_or_two_line_minified_candidate",
            "pathological_line_repetition_candidate",
            "license_or_comment_only_candidate",
            *PATH_RULES,
        ):
            if evidence[key]:
                candidate_counts[key] += 1
                candidate_tokens[key] += token_proxy

    provenance_backed = {
        "declared_generated": metadata_values["artifact_context.generation"].get("generated", 0),
        "declared_dependency_copy": metadata_values["artifact_context.dependency_copy"].get("true", 0),
    }
    rule_assessment = {
        "declared_generated_artifact": {
            "evidence": "artifact_context.generation=generated",
            "candidates": provenance_backed["declared_generated"],
            "status": "needs_executable_false_positive_fixture" if provenance_backed["declared_generated"] else "blocked_missing_input_metadata",
        },
        "declared_dependency_copy": {
            "evidence": "artifact_context.dependency_copy=true",
            "candidates": provenance_backed["declared_dependency_copy"],
            "status": "needs_executable_false_positive_fixture" if provenance_backed["declared_dependency_copy"] else "blocked_missing_input_metadata",
        },
        "path_based_vendor_or_generated": {
            "evidence": "relative path pattern only",
            "candidates": sum(candidate_counts[key] for key in PATH_RULES),
            "status": "blocked_known_false_positive_risk",
            "reason": "A path can identify a useful dependency or authored generated source; it is not source-declared artifact context.",
        },
        "text_shape_only": {
            "evidence": "line shape or repetition only",
            "candidates": sum(candidate_counts[key] for key in ("one_or_two_line_minified_candidate", "pathological_line_repetition_candidate")),
            "status": "blocked_known_false_positive_risk",
            "reason": "Chunk boundaries can create long numeric lines and repeated legitimate code; an executable false-positive fixture is required before selection use.",
        },
    }
    return {
        "schema_version": "core-rule-opportunity-audit-v1",
        "status": "diagnostic_only_no_new_selection_policy",
        "scope": "Current materialized corpus only. Candidate counts do not measure intrinsic quality or authorize removal.",
        "corpus": {"chunks": total_chunks, "token_proxy": total_tokens},
        "metadata_availability": {
            "present_chunks": dict(metadata_present),
            "declared_value_counts": {key: dict(values) for key, values in metadata_values.items()},
        },
        "observable_candidates": {
            key: {"chunks": candidate_counts[key], "token_proxy": candidate_tokens[key]}
            for key in (
                "strong_generated_marker_candidate",
                "one_or_two_line_minified_candidate",
                "pathological_line_repetition_candidate",
                "license_or_comment_only_candidate",
                *PATH_RULES,
            )
        },
        "rule_assessment": rule_assessment,
        "next_input_contract_requirement": "Collectors must declare artifact_context.generation and artifact_context.dependency_copy from source-backed metadata when available. Neither may be inferred from a path alone.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit evidence available for additional Core selection rules.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(_read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "corpus": report["corpus"], "rule_assessment": report["rule_assessment"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
