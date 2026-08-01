from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Final, Literal


ScopeStatus = Literal["in_scope", "out_of_scope_abstain"]
SUPPORTED_LANGUAGE_DECLARATIONS: Final = frozenset({"source_row", "user_declared"})


@dataclass(frozen=True, slots=True)
class CodeStructuralEvidence:
    status: ScopeStatus
    route_confidence: float
    substantive_payload: float
    coherence_completeness: float
    reason_code: str


def _is_non_payload_statement(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and (isinstance(statement.value.value, str) or statement.value.value is Ellipsis)
    )


def inspect_python_complete_source(
    text: str,
    language_code: str,
    language_declaration: str | None,
    record_shape: str,
) -> CodeStructuralEvidence:
    """Build text-structural heads only for complete, declared Python sources."""
    in_scope = (
        language_code.casefold() == "python"
        and language_declaration in SUPPORTED_LANGUAGE_DECLARATIONS
        and record_shape == "complete_source"
    )
    if not in_scope:
        return CodeStructuralEvidence(
            "out_of_scope_abstain",
            0.0,
            0.0,
            0.0,
            "python_complete_source_scope_not_established_abstain",
        )
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return CodeStructuralEvidence(
            "in_scope",
            1.0,
            0.0,
            0.0,
            "python_complete_source_parse_failed_abstain",
        )
    substantive = any(not _is_non_payload_statement(statement) for statement in tree.body)
    return CodeStructuralEvidence(
        "in_scope",
        1.0,
        float(substantive),
        1.0,
        "python_complete_source_structural_evidence_built",
    )
