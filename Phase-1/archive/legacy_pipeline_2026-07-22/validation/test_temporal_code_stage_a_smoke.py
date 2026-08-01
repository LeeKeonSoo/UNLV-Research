#!/usr/bin/env python3
"""Regression checks for split-isolated temporal-code Stage-A smoke."""

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record(record_id: str, split: str, text: str) -> dict:
    return {
        "record_id": record_id,
        "text": text,
        "partition": {
            "split": split,
            "bundle_id": f"{split}-bundle",
            "repository_identity": f"fixture/{split}",
            "path": "src/example.py",
            "content_type": "code",
        },
    }


def main() -> int:
    module = importlib.import_module("74_run_temporal_code_stage_a_smoke")
    duplicate = "def useful(value):\n    return value + 1\n"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write(root / "train" / "release_candidates.jsonl", [_record("train-a", "train", duplicate)])
        _write(
            root / "development" / "release_candidates.jsonl",
            [_record("dev-a", "development", duplicate), _record("dev-b", "development", duplicate)],
        )
        _write(root / "confirmatory" / "release_candidates.jsonl", [_record("confirm-a", "confirmatory", duplicate)])
        report = module.run(root, root / "out")
    splits = report["summary"]["split_counts"]
    assert splits["train"]["stage_a_pass_count"] == 1, splits
    assert splits["development"]["stage_a_rejected_count"] == 1, splits
    assert splits["confirmatory"]["stage_a_pass_count"] == 1, splits
    assert "Utility" in report["stage_a_contract"]["forbidden_signals"], report
    print("[temporal-code-stage-a] split-isolated duplicate decisions: pass")
    print("[temporal-code-stage-a] Utility excluded from hard-gate contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
