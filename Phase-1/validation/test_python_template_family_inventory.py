#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from scripts.build_python_template_family_inventory import build_inventory

    rows = [
        {"record_id": "one", "text": "def add(value):\n    return value + 1\n", "language": {"code": "python"}, "partition": {"path": "one.py"}},
        {"record_id": "two", "text": "def increment(item):\n    return item + 2\n", "language": {"code": "python"}, "partition": {"path": "two.py"}},
        {"record_id": "three", "text": "def negate(value):\n    return -value\n", "language": {"code": "python"}, "partition": {"path": "three.py"}},
    ]
    report = build_inventory(rows, minimum_tokens=8, sample_limit=5)
    assert report["duplicate_family_count"] == 1
    assert report["records_in_duplicate_families"] == 2
    assert report["family_samples"][0]["family_size"] == 2
    print("[python-template-family] alpha-normalized family inventory: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
