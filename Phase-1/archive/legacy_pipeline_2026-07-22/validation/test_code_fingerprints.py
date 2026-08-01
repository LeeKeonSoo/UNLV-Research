#!/usr/bin/env python3
"""Regression checks for derived benchmark near-duplicate fingerprints."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_fingerprints import derived_fingerprints, simhash_hamming_distance  # noqa: E402
from ingestion.temporal_code_manifests import benchmark_quarantine_decision  # noqa: E402


def main() -> int:
    first = derived_fingerprints("def add(left, right):\n    return left + right\n")
    renamed = derived_fingerprints("def add(x, y):\n    return x + y\n")
    unrelated = derived_fingerprints("class Runner:\n    def stop(self):\n        raise RuntimeError()\n")
    assert first["python_ast_sha256"] == renamed["python_ast_sha256"], (first, renamed)
    assert first["python_ast_sha256"] != unrelated["python_ast_sha256"], (first, unrelated)
    assert simhash_hamming_distance(first["token_simhash64"], first["token_simhash64"]) == 0
    cross_repository = benchmark_quarantine_decision(
        {
            "repository_identity": "different/copied-repo",
            "parent_commit": "1" * 40,
            "merge_commit": "2" * 40,
            "content_signatures": [{"normalized_sha256": "a" * 64, **first}],
        },
        {
            "entries": [
                {
                    "benchmark": "FixtureBench",
                    "repository_patterns": [],
                    "text_sha256": [],
                    "task_artifact_rules": [
                        {
                            "repository_identity": "benchmark/original-repo",
                            "commit_oids": [],
                            "normalized_sha256": ["a" * 64],
                            "token_simhash64": [first["token_simhash64"]],
                            "python_ast_sha256": [first["python_ast_sha256"]],
                        }
                    ],
                }
            ]
        },
    )
    reasons = set(cross_repository["matches"][0]["reasons"])
    assert reasons == {
        "benchmark_task_content_hash",
        "benchmark_task_ast_structure_hash",
        "benchmark_task_token_simhash_near_duplicate",
    }, reasons
    print("[code-fingerprints] normalized AST and token SimHash derivation: pass")
    print("[code-fingerprints] cross-repository benchmark artifact quarantine: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
