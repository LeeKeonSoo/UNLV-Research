#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.official_suite_generator import (
    bigcodebench_parquet_path,
    livecodebench_release_files,
    output_path,
)


def main() -> int:
    run_root = Path("D:/runs")
    actual = output_path(run_root, "ds1000", "curated_natural", 23)
    expected = run_root / "official_suite_samples" / "ds1000_curated_natural_seed23.jsonl"
    assert actual == expected

    fixture_root = Path("D:/fixture")
    expected_lcb = [
        fixture_root / "test.jsonl",
        fixture_root / "test2.jsonl",
        fixture_root / "test3.jsonl",
        fixture_root / "test4.jsonl",
        fixture_root / "test5.jsonl",
        fixture_root / "test6.jsonl",
    ]
    assert livecodebench_release_files(fixture_root, "release_v6") == expected_lcb
    assert bigcodebench_parquet_path(fixture_root) == fixture_root / "data" / "v0.1.4-00000-of-00001.parquet"
    print("[official-suite-generator] output path contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
