#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def main() -> int:
    module = importlib.import_module("157_freeze_code_domain_v2_stage_b_arms")
    fingerprint = module._implementation_sha256()
    assert set(fingerprint) == {"ingestion/code_chunks.py", "ingestion/code_selection.py"}
    assert all(len(value) == 64 for value in fingerprint.values())
    print("[stage-b-implementation-fingerprint] current implementation hashes: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
