#!/usr/bin/env python3
"""Ordered entrypoint: validate outputs and gates."""

from pathlib import Path

from phase1_autosave import autosave_stage
from validate_phase1_outputs import main

PROJECT_DIR = Path(__file__).resolve().parent


def _main_with_autosave() -> int:
    rc = main()
    if rc == 0:
        try:
            autosave_stage(PROJECT_DIR, "validate_phase1_outputs")
        except Exception as e:
            print(f"[Autosave] warning (07): {e}")
    return rc


if __name__ == "__main__":
    raise SystemExit(_main_with_autosave())
