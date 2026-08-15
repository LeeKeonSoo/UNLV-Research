#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.evalplus_windows_runner import (
    install_windows_signal_compatibility,
)


def test_windows_compatibility_preserves_pass_and_fail_execution() -> None:
    if os.name != "nt":
        return
    install_windows_signal_compatibility()
    from evalplus.eval import PASS, untrusted_check

    passing_code = "def add_one(value):\n    return value + 1\n"
    failing_code = "def add_one(value):\n    return value\n"
    common = {
        "dataset": "humaneval",
        "inputs": [[1]],
        "entry_point": "add_one",
        "expected": [2],
        "atol": 0.0,
        "ref_time": [0.001],
        "fast_check": True,
    }
    passing_status, _ = untrusted_check(code=passing_code, **common)
    failing_status, _ = untrusted_check(code=failing_code, **common)
    assert passing_status == PASS
    assert failing_status != PASS


if __name__ == "__main__":
    test_windows_compatibility_preserves_pass_and_fail_execution()
    print("[evalplus-windows-runner] pass/fail execution semantics: pass")
