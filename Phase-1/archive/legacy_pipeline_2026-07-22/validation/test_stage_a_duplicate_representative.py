#!/usr/bin/env python3
"""Validate deterministic eligible-only Stage-A duplicate representatives."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.code_chunks import apply_stage_a_hard_gates  # noqa: E402


def _by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["chunk_uid"]): row for row in rows}


def main() -> int:
    shared = "def normalize(value):\n    cleaned = value.strip().lower()\n    return cleaned\n"
    records = [
        {
            "chunk_uid": "z-invalid-doc",
            "path": "notes/short.txt",
            "content_type": "documentation",
            "split": "train",
            "text": shared,
        },
        {
            "chunk_uid": "b-valid-copy",
            "path": "src/copy.py",
            "content_type": "code",
            "split": "train",
            "text": shared,
        },
        {
            "chunk_uid": "a-valid-representative",
            "path": "src/original.py",
            "content_type": "code",
            "split": "train",
            "text": shared,
        },
    ]
    expected = None
    for permutation in itertools.permutations(records):
        by_id = _by_id(apply_stage_a_hard_gates(list(permutation)))
        snapshot = {
            uid: {
                "pass": row["stage_a_pass"],
                "blockers": row["stage_a_blockers"],
                "exact_match": row["exact_duplicate_match"],
                "eligible": row["duplicate_representative_eligible"],
            }
            for uid, row in by_id.items()
        }
        expected = expected or snapshot
        assert snapshot == expected, (snapshot, expected)
    assert expected is not None
    assert expected["a-valid-representative"]["pass"] is True
    assert expected["b-valid-copy"]["exact_match"] == "a-valid-representative"
    assert "exact_duplicate_within_split" in expected["b-valid-copy"]["blockers"]
    assert expected["z-invalid-doc"]["eligible"] is False
    assert expected["z-invalid-doc"]["blockers"] == ["below_minimum_learnable_unit"]
    print("[stage-a-representative] local-invalid chunks cannot suppress usable representatives: pass")
    print("[stage-a-representative] decisions are invariant to input order: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
