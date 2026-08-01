#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code_positive_evidence import inspect_python_complete_source


def test_complete_python_implementation_has_independent_structural_heads() -> None:
    # Given: a complete source with an authoritative Python declaration.
    text = "def add(left, right):\n    return left + right\n"

    # When: the structural Code route is evaluated.
    result = inspect_python_complete_source(text, "python", "source_row", "complete_source")

    # Then: route, payload, and coherence are independently evidenced.
    assert result.status == "in_scope"
    assert result.route_confidence == 1.0
    assert result.substantive_payload == 1.0
    assert result.coherence_completeness == 1.0


def test_interface_definition_is_substantive_payload() -> None:
    # Given: a valid interface whose body intentionally contains no implementation.
    text = "class Adapter:\n    def connect(self):\n        pass\n"

    # When: the source is structurally inspected.
    result = inspect_python_complete_source(text, "python", "source_row", "complete_source")

    # Then: the declared API remains positive payload evidence.
    assert result.substantive_payload == 1.0
    assert result.coherence_completeness == 1.0


def test_comment_docstring_and_pass_only_source_has_no_substantive_payload() -> None:
    # Given: complete Python syntax with no declaration or executable payload.
    text = '"""License notice."""\npass\n'

    # When: the source is structurally inspected.
    result = inspect_python_complete_source(text, "python", "source_row", "complete_source")

    # Then: parsing succeeds but substantive payload remains independently absent.
    assert result.substantive_payload == 0.0
    assert result.coherence_completeness == 1.0


def test_syntax_error_and_unsupported_scope_abstain_without_rejection() -> None:
    # Given: one malformed declared Python source and two unsupported inputs.
    malformed = inspect_python_complete_source("def broken(:\n", "python", "source_row", "complete_source")
    unknown = inspect_python_complete_source("def ok():\n    return 1\n", "und", None, "complete_source")
    snippet = inspect_python_complete_source("return value", "python", "source_row", "snippet")

    # When/Then: malformed code lacks coherence; unsupported scopes abstain.
    assert malformed.status == "in_scope"
    assert malformed.coherence_completeness == 0.0
    assert malformed.reason_code == "python_complete_source_parse_failed_abstain"
    assert unknown.status == "out_of_scope_abstain"
    assert snippet.status == "out_of_scope_abstain"


if __name__ == "__main__":
    test_complete_python_implementation_has_independent_structural_heads()
    test_interface_definition_is_substantive_payload()
    test_comment_docstring_and_pass_only_source_has_no_substantive_payload()
    test_syntax_error_and_unsupported_scope_abstain_without_rejection()
    print("[code-positive-evidence] independent structural heads and abstention: pass")
