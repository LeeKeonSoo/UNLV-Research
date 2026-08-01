#!/usr/bin/env python3
"""Build candidate-only structural evidence for complete declared Python sources."""
from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


JsonMap = dict[str, Any]
SYNTAX_ERROR_REASON = "python_syntax_error_source_candidate"
NON_EXECUTABLE_STUB_REASON = "python_non_executable_stub_source_candidate"


def _is_docstring_or_ellipsis(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and (isinstance(statement.value.value, str) or statement.value.value is Ellipsis)
    )


def _is_exception_class(node: ast.ClassDef) -> bool:
    return any(
        isinstance(base, ast.Name) and base.id.endswith(("Error", "Exception", "Warning"))
        for base in node.bases
    )


def _is_stub_body(statements: list[ast.stmt]) -> bool:
    return bool(statements) and all(
        isinstance(statement, ast.Pass)
        or _is_docstring_or_ellipsis(statement)
        or (
            isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _is_stub_body(statement.body)
        )
        or (
            isinstance(statement, ast.ClassDef)
            and not _is_exception_class(statement)
            and _is_stub_body(statement.body)
        )
        for statement in statements
    )


def _parse_feature_version(language_version: str) -> tuple[int, int] | None:
    """Return a CPython grammar version only when the input declares Python 3.x."""
    pieces = language_version.split(".")
    if len(pieces) < 2 or pieces[0] != "3" or not pieces[1].isdigit():
        return None
    return 3, int(pieces[1])


def _candidate_reason(text: str, feature_version: tuple[int, int]) -> str | None:
    """Return only direct syntax or whole-source stub evidence; never a quality score."""
    try:
        tree = ast.parse(text, feature_version=feature_version)
    except SyntaxError:
        return SYNTAX_ERROR_REASON
    return NON_EXECUTABLE_STUB_REASON if _is_stub_body(tree.body) else None


def analyze_python_source_records(rows: Iterable[JsonMap]) -> JsonMap:
    """Audit declared Python sources without authorizing selection or removal."""
    counts: dict[str, int] = defaultdict(int)
    candidates: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        language = row.get("language") if isinstance(row.get("language"), dict) else {}
        if language.get("code") != "python":
            counts["non_python"] += 1
            continue
        counts["python_records"] += 1
        feature_version = _parse_feature_version(str(language.get("version") or ""))
        if feature_version is None:
            counts["version_unresolved_not_evaluated"] += 1
            continue
        reason = _candidate_reason(str(row["text"]), feature_version)
        if reason is None:
            counts["retained_without_candidate_evidence"] += 1
            continue
        match reason:
            case matched_reason if matched_reason == SYNTAX_ERROR_REASON:
                counts["syntax_error"] += 1
            case matched_reason if matched_reason == NON_EXECUTABLE_STUB_REASON:
                counts["non_executable_stub_candidate"] += 1
            case unreachable:
                raise RuntimeError(f"Unexpected code evidence reason: {unreachable}")
        candidates[reason].append(str(row["record_id"]))
    return {
        "schema_version": "python-code-evidence-audit-v1",
        "status": "candidate_evidence_only_not_a_runtime_selection_policy",
        "scope": "Complete sources explicitly declared as Python 3.x. Python 2, missing, or unsupported versions are not evaluated. Stub-only modules can be intentional interfaces. Both candidate reasons require false-positive review and external validation before any promotion.",
        "forbidden_inputs": ["intrinsic_quality_score", "human_quality_label", "Utility", "NLL", "benchmark_outcomes", "source_identity", "source_tier"],
        "counts": dict(counts),
        "candidate_record_ids": dict(candidates),
    }
