#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic_coverage_corpus_runner import (
    audit_corpus,
    encode_provider,
    load_run_config,
    prepare_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hash-linked semantic Coverage evidence")
    parser.add_argument("command", choices=("prepare", "encode", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", choices=("primary", "audit"))
    args = parser.parse_args()
    config = load_run_config(args.config)
    if args.command == "prepare":
        result = prepare_corpus(config)
    elif args.command == "encode":
        if args.provider is None:
            parser.error("encode requires --provider")
        result = encode_provider(config, args.provider)
    else:
        result = audit_corpus(config)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
