#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.code_chunks import apply_stage_a_hard_gates, hard_near_duplicate_evidence, stage_a_local_evidence, token_shingles  # noqa: E402


def main() -> int:
    base = " ".join(f"feature_{index}" for index in range(80))
    variant = " ".join([*(f"feature_{index}" for index in range(79)), "replacement_feature"])
    evidence = hard_near_duplicate_evidence(
        {**stage_a_local_evidence({"text": base}), "shingles": token_shingles(base)},
        {**stage_a_local_evidence({"text": variant}), "shingles": token_shingles(variant)},
    )
    assert evidence["match"] is True, evidence
    decisions = apply_stage_a_hard_gates(
        [
            {"chunk_uid": "a", "path": "notes/a.txt", "content_type": "documentation", "split": "train", "text": base},
            {"chunk_uid": "b", "path": "notes/b.txt", "content_type": "documentation", "split": "train", "text": variant},
        ]
    )
    by_id = {row["chunk_uid"]: row for row in decisions}
    assert by_id["a"]["stage_a_pass"] is True
    assert by_id["b"]["stage_a_pass"] is False
    assert "hard_near_duplicate_within_split" in by_id["b"]["stage_a_blockers"]
    assert by_id["b"]["hard_near_duplicate_match"] == "a"
    print("[stage-a-hard-near-duplicate] detected hard near duplicates are rejected: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
