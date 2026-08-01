#!/usr/bin/env python3
"""Regression checks for pure-formatting collection exclusion."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ingestion.code_change import substantive_change_decision  # noqa: E402


def main() -> int:
    formatting = substantive_change_decision(
        {
            "path": "src/example.py",
            "change_type": "modified",
            "before_text": "value=1\n",
            "after_text": "value = 1\n",
        }
    )
    assert formatting == {"substantive": False, "method": "python_ast_without_locations"}, formatting
    logic = substantive_change_decision(
        {
            "path": "src/example.py",
            "change_type": "modified",
            "before_text": "value = 1\n",
            "after_text": "value = 2\n",
        }
    )
    assert logic["substantive"] is True, logic
    unavailable = substantive_change_decision(
        {"path": "src/private.py", "change_type": "modified", "before_text": None, "after_text": None}
    )
    assert unavailable["substantive"] is None, unavailable
    print("[substantive-change] Python formatting-only and logic-change separation: pass")
    print("[substantive-change] unavailable content remains unverified: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
