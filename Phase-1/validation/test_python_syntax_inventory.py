#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from scripts.build_python_syntax_inventory import build_inventory

    report = build_inventory(
        [
            {"record_id": "valid", "text": "def add(left, right):\n    return left + right\n", "language": {"code": "python"}},
            {"record_id": "invalid", "text": "def broken(:\n    pass\n", "language": {"code": "python"}, "partition": {"path": "broken.py"}},
            {"record_id": "other", "text": "const value = 1;", "language": {"code": "javascript"}},
        ],
        sample_limit=5,
    )
    assert report["counts"] == {"python_records": 2, "parseable": 1, "syntax_error": 1, "non_python": 1}
    assert report["syntax_error_categories"] == {"ambiguous_syntax_error": 1}
    assert report["syntax_error_samples"]["ambiguous_syntax_error"][0]["path"] == "broken.py"
    print("[python-syntax-inventory] source-level parseability accounting: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
