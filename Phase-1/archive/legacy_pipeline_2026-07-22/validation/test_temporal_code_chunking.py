#!/usr/bin/env python3
"""Regression checks for syntax-aware temporal-code chunking and Stage A."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_chunks import apply_stage_a_hard_gates, python_syntax_chunks, syntax_aware_chunks  # noqa: E402


def main() -> int:
    source = '''"""module docs"""
import os

def first(value):
    return value + 1

class Runner:
    def run(self):
        return True
'''
    chunks = python_syntax_chunks(source)
    assert [row["kind"] for row in chunks] == ["module_statements", "function", "class"], chunks
    assert all(ast.parse(row["text"]) for row in chunks), chunks

    broken = syntax_aware_chunks(
        {"text": "def broken(:\n", "partition": {"path": "broken.py", "content_type": "code"}}
    )
    assert broken["parseable"] is False, broken

    base = {
        "path": "src/example.py",
        "content_type": "code",
        "split": "train",
    }
    decisions = apply_stage_a_hard_gates(
        [
            {**base, "chunk_uid": "a", "text": "def add(left, right):\n    return left + right\n"},
            {**base, "chunk_uid": "b", "text": "def add(x, y):\n    return x + y\n"},
            {**base, "chunk_uid": "c", "text": "def add(left, right):\n    return left + right\n\n"},
            {**base, "chunk_uid": "d", "text": "def add(left, right):\n    return left - right\n"},
        ]
    )
    assert decisions[0]["stage_a_pass"] is True, decisions
    assert decisions[1]["stage_a_pass"] is True, decisions
    assert "exact_duplicate_within_split" in decisions[2]["stage_a_blockers"], decisions
    assert decisions[2]["exact_duplicate_match"] == "a", decisions
    assert decisions[3]["stage_a_pass"] is True, decisions
    assert "hard_near_duplicate_within_split" not in decisions[3]["stage_a_blockers"], decisions
    assert decisions[2]["duplicate_representative_policy"] == "local_gate_pass_then_canonical_exact_lexicographic_v2"
    print("[temporal-code-chunking] top-level syntax boundaries remain parseable: pass")
    print("[temporal-code-chunking] parseability and canonical-exact Stage-A gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
