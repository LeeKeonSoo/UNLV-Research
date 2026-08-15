#!/usr/bin/env python3
"""Run EvalPlus on Windows while retaining its process-level task timeout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
from typing import Any


def install_windows_signal_compatibility() -> None:
    """Provide the Unix timer names EvalPlus expects inside spawned workers."""
    if os.name != "nt":
        return
    if not hasattr(signal, "ITIMER_REAL"):
        signal.ITIMER_REAL = 0  # type: ignore[attr-defined]
    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = signal.SIGBREAK  # type: ignore[attr-defined]
    if not hasattr(signal, "setitimer"):
        def setitimer_compat(*_args: Any, **_kwargs: Any) -> tuple[float, float]:
            return (0.0, 0.0)

        signal.setitimer = setitimer_compat  # type: ignore[attr-defined]
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"


install_windows_signal_compatibility()


def main() -> int:
    from evalplus.evaluate import evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("humaneval", "mbpp"))
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--test-details", action="store_true")
    args = parser.parse_args()
    evaluate(
        dataset=args.dataset,
        samples=str(args.samples),
        parallel=args.parallel,
        i_just_wanna_run=True,
        test_details=args.test_details,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
