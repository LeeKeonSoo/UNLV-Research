#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.preflight_code_7benchmark_pretraining_eligible_v3 import preflight


def main() -> int:
    report = preflight()
    assert report["status"] == "preflight_ready_with_declared_blocks"
    assert len(report["checked_files"]) >= 12
    assert report["pending_gates"] == [
        "v3 adapter training has not been materialized",
        "Qwen3-4B pretraining cutoff lacks an auditable declaration",
        "raw corpus snapshot end lacks an auditable declaration",
    ]
    print("[code-7benchmark-preflight] frozen artifacts and declared blocks: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
