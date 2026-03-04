#!/usr/bin/env python3
"""Ordered entrypoint: build dashboard HTML."""

from pathlib import Path

from phase1_autosave import autosave_stage
from build_dashboard import main

PROJECT_DIR = Path(__file__).resolve().parent


def _main_with_autosave() -> int:
    rc = main()
    if rc == 0:
        try:
            autosave_stage(PROJECT_DIR, "build_dashboard")
        except Exception as e:
            print(f"[Autosave] warning (06): {e}")
    return rc


if __name__ == "__main__":
    raise SystemExit(_main_with_autosave())
