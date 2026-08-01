#!/usr/bin/env python3
"""Regression checks for score-hidden temporal-code Stage-B review packets."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    module = importlib.import_module("78_build_temporal_code_stage_b_blind_review")
    rows = []
    for index in range(30):
        content_type = ("code", "test", "documentation")[index % 3]
        rows.append(
            {
                "chunk_uid": f"chunk-{index:03d}",
                "repository_identity": "fixture/repo",
                "bundle_id": f"bundle-{index % 2}",
                "path": f"src/item_{index}.py",
                "content_type": content_type,
                "change_type": "modified",
                "chunk_kind": "function",
                "text": f"def item_{index}(value):\n    return value + {index}\n",
                "stage_b_evidence": {
                    "stage_b_objective_score": index / 30,
                    "soft_redundancy_risk": (index % 5) / 5,
                    "pass_through_assignment_ratio": 0.5 if index % 7 == 0 else 0.0,
                },
            }
        )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write(root / "scored.jsonl", rows)
        _write(root / "selected.jsonl", rows[:10])
        _write(root / "baseline.jsonl", rows[10:20])
        packet = module.build(root / "scored.jsonl", root / "selected.jsonl", root / "baseline.jsonl", root / "out")
        key = json.loads((root / "out" / "blind_review_key.json").read_text(encoding="utf-8"))
    forbidden = {"chunk_uid", "repository_identity", "bundle_id", "path", "arm", "stage_b_evidence", "sampling_stratum", "stratum"}
    required_key = {"chunk_uid", "repository_identity", "bundle_id", "path", "arm", "stage_b_evidence", "sampling_stratum"}
    assert packet["status"] == "awaiting_independent_review", packet
    assert all(not forbidden.intersection(row) for row in packet["records"]), packet
    assert all(required_key.issubset(row) for row in key["records"]), key
    assert packet["review_contract"]["scores_and_selection_arms_hidden"] is True, packet
    assert packet["review_contract"]["sampling_strata_hidden"] is True, packet
    print("[temporal-code-stage-b-blind-review] scores and arms hidden from packet: pass")
    print("[temporal-code-stage-b-blind-review] separate key retains audit lineage: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
