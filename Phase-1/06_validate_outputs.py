#!/usr/bin/env python3
"""Canonical entrypoint: validate outputs for the generic data evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from validate_outputs import VALIDATION_REPORT_PATH, main as validate_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data evaluation outputs.")
    parser.add_argument("--write-report", type=Path, default=VALIDATION_REPORT_PATH)
    args = parser.parse_args()
    return validate_main(write_report=args.write_report)


if __name__ == "__main__":
    raise SystemExit(main())
