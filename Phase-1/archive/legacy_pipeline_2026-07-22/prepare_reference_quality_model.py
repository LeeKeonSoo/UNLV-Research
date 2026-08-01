#!/usr/bin/env python3
"""Prepare a reference-trained quality classifier."""

from __future__ import annotations

import argparse
import json
from typing import List

from datasets import load_dataset

from quality.reference_quality import (
    build_reference_quality_model,
    save_reference_quality_model,
    tokenize_quality_text,
)


DEFAULT_DATASET_NAME = "Salesforce/wikitext"
DEFAULT_CONFIG_NAME = "wikitext-2-raw-v1"
DEFAULT_SPLIT = "train"
DEFAULT_TOKEN_BUDGET = 2_000_000
DEFAULT_MAX_TEXTS = 50_000
DEFAULT_N_FEATURES = 2**18
DEFAULT_SEED = 42


def _is_reference_line(text: str) -> bool:
    stripped = " ".join(text.split()).strip()
    if len(stripped) < 80:
        return False
    if stripped.startswith("=") and stripped.endswith("="):
        return False
    if stripped.lower().startswith(("see also", "external links", "references")):
        return False
    return True


def _collect_reference_texts(
    dataset_name: str,
    config_name: str,
    split: str,
    *,
    token_budget: int,
    max_texts: int,
) -> List[str]:
    ds = load_dataset(dataset_name, config_name, split=split)
    texts: List[str] = []
    token_total = 0
    for row in ds:
        raw = str(row.get("text") or "")
        if not _is_reference_line(raw):
            continue
        tokens = tokenize_quality_text(raw)
        if len(tokens) < 8:
            continue
        texts.append(raw)
        token_total += len(tokens)
        if len(texts) >= max_texts or token_total >= token_budget:
            break
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a reference quality classifier.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    parser.add_argument("--max-texts", type=int, default=DEFAULT_MAX_TEXTS)
    parser.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    texts = _collect_reference_texts(
        args.dataset_name,
        args.config_name,
        args.split,
        token_budget=args.token_budget,
        max_texts=args.max_texts,
    )
    if not texts:
        raise SystemExit("No reference texts were collected for the quality model.")

    model = build_reference_quality_model(
        texts,
        n_features=args.n_features,
        reference_source=f"{args.dataset_name}:{args.config_name}:{args.split}",
        metadata={
            "dataset_name": args.dataset_name,
            "config_name": args.config_name,
            "split": args.split,
            "token_budget": args.token_budget,
            "max_texts": args.max_texts,
        },
        seed=args.seed,
    )
    save_reference_quality_model(model)
    print("[quality-ref] prepared reference quality model")
    print(json.dumps(model.metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
