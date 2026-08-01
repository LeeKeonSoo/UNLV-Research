#!/usr/bin/env python3
"""Canonical entrypoint: build index artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import DEFAULT_DATASET_CONFIG
from index.build import build_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build index artifacts for generic data evaluation.")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    args = parser.parse_args()
    manifest = build_index(args.datasets_config)
    print(f"[02] index manifest: {manifest['index_db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
