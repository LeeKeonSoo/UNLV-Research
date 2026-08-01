#!/usr/bin/env python3
"""Regression checks for split-preserving temporal-code Stage-0 adaptation."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    adapter = importlib.import_module("73_prepare_temporal_code_stage0_candidates")
    bundle_path = PROJECT_DIR / "validation" / "fixtures" / "temporal_code_change_bundles.json"
    fixture = load_json(bundle_path)["bundles"][0]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / "bundle.json"
        temporary.write_text(json.dumps(fixture), encoding="utf-8")
        audit = {
            "decisions": [
                {
                    "stage0_release_candidate": True,
                    "assigned_split": "train",
                    "bundle_path": str(temporary),
                }
            ]
        }
        rows = list(adapter.raw_candidates(audit))
    assert len(rows) == 2, rows
    assert all(row["partition"]["split"] == "train" for row in rows), rows
    assert {row["partition"]["change_type"] for row in rows} == {"added", "modified"}, rows
    assert all(row["text"] != fixture["prose"]["body"] for row in rows), rows
    assert all(row["pii_context"] == "repository_code" for row in rows), rows
    print("[temporal-code-stage0-adapter] split preservation and prose exclusion: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
