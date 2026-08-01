#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from validate_outputs import includes_historical_evidence  # noqa: E402


def main() -> int:
    assert includes_historical_evidence("canonical") is False
    assert includes_historical_evidence("full") is True
    print("[validation-scope] canonical excludes historical temporal-code evidence: pass")
    print("[validation-scope] full retains historical temporal-code evidence: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
