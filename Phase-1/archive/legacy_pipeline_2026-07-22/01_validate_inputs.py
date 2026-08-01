#!/usr/bin/env python3
"""Validate prepared dataset inputs for the generic data evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import DEFAULT_DATASET_CONFIG, iter_documents, normalize_dataset_config


def validate_inputs(dataset_config: Path) -> List[str]:
    specs = normalize_dataset_config(dataset_config)
    messages: List[str] = []
    for spec in specs:
        source = Path(spec["source"])
        if not source.exists():
            raise FileNotFoundError(f"{spec['name']}: source not found: {source}")
        seen = 0
        text_field = spec["text_field"]
        for row in iter_documents(spec):
            doc = row["doc"]
            if text_field not in doc:
                raise ValueError(f"{spec['name']}: missing text field '{text_field}' in sampled document")
            seen += 1
            if seen >= 3:
                break
        if seen == 0:
            raise ValueError(f"{spec['name']}: no readable documents found in source")
        messages.append(f"{spec['name']}: OK ({seen} sampled docs)")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prepared dataset inputs.")
    parser.add_argument("--datasets-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    args = parser.parse_args()

    messages = validate_inputs(args.datasets_config)
    for msg in messages:
        print(f"[01] {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
