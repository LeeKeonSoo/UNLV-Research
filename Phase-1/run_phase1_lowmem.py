#!/usr/bin/env python3
"""Run Phase-1 pipeline with a memory-sensitive profile and strict validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from validate_phase1_outputs import (
    ValidationItem,
    validate_analysis_outputs,
    validate_corpus_index,
    validate_dashboard,
    validate_khan_collection,
    validate_manifest,
    validate_taxonomy_outputs,
    validate_tiny_collection,
)


PROJECT_DIR = Path(__file__).resolve().parent


def _run(cmd: List[str], env: Dict[str, str], label: str) -> None:
    """Run a command and fail fast with a readable error."""
    print(f"\n[Phase-1] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=env)
    if result.returncode != 0:
        raise SystemExit(f"[Phase-1] failed at: {label} (code {result.returncode})")


def _print_results(results: Sequence[ValidationItem], context: str) -> None:
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n[Phase-1] {context} validation failed:")
        for item in failed:
            print(f"  [FAIL] {item.name}: {item.details}")
        raise SystemExit(f"{context} validation failed.")

    passed = [r for r in results if r.ok]
    if not passed:
        return
    sample_msg = []
    for item in passed:
        if item.details:
            sample = str(item.details)
            sample_msg.append(f"{item.name} ✅ {sample}")
        else:
            sample_msg.append(f"{item.name} ✅")
    print(f"[Phase-1] {context} checks: OK ({len(passed)})")
    for s in sample_msg[:3]:
        print(f"    - {s}")


def _run_and_validate(
    label: str,
    cmd: List[str],
    env: Dict[str, str],
    checks: Sequence[Callable[[], List[ValidationItem]]],
) -> None:
    _run(cmd, env, label)
    for check in checks:
        _print_results(check(), f"{label}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-1 pipeline with strict validation")
    parser.add_argument(
        "--max-tiny-batches",
        type=int,
        default=None,
        help="Optional tiny-textbooks batch-limit for collection validation.",
    )
    parser.add_argument(
        "--max-metrics-records",
        type=int,
        default=None,
        help="Optional metrics JSONL record limit for validation after compute step.",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Fail if run_manifest reliability gates have pass=False.",
    )
    parser.add_argument(
        "--rerun-all",
        action="store_true",
        help="Re-run every stage even if outputs already exist.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable batching in compute/index stages before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Tuned low-memory profile defaults.
    profile: Dict[str, str] = {
        "PHASE1_DEVICE": "auto",
        "PHASE1_CUDA_DEVICE": "0",
        "PHASE1_USE_GPU": "1",
        "PHASE1_DOMAIN_BATCH_SIZE": "256",
        "PHASE1_MAX_BATCHES": "1",
        "PHASE1_INDEX_MAX_BATCHES": "1",
        "PHASE1_TFIDF_MAX_FEATURES": "2000",
        "PHASE1_BUILD_TFIDF_MATRIX": "1",
        "PHASE1_STORE_DOC_TEXTS": "0",
        "PHASE1_ENABLE_MINHASH": "0",
        "PHASE1_QUERY_CACHE_LIMIT": "256",
        "PHASE1_NGRAM_CACHE_LIMIT": "512",
        "PHASE1_SEMANTIC_CANDIDATE_LIMIT": "80",
        "PHASE1_NGRAM_CANDIDATE_LIMIT": "40",
        "PHASE1_SKIP_REDUNDANCY": "1",
        "PHASE1_SKIP_PERPLEXITY": "0",
    }

    if args.full:
        profile["PHASE1_MAX_BATCHES"] = "0"
        profile["PHASE1_INDEX_MAX_BATCHES"] = "0"

    env = os.environ.copy()
    env.update(profile)

    print("[Phase-1] Using low-memory profile:")
    for k in sorted(profile):
        print(f"  {k}={env[k]}")

    khan_path = PROJECT_DIR / "khan_k12_concepts" / "all_k12_concepts.json"
    tiny_dir = PROJECT_DIR / "tiny_textbooks_raw"
    outputs = PROJECT_DIR / "outputs"
    concept_out = outputs / "concept_prototypes_tfidf.pkl"

    # Always validate pre-existing inputs before deciding which stages to skip.
    if khan_path.exists() and not args.rerun_all:
        _print_results(validate_khan_collection(max_records=args.max_tiny_batches), "Khan collection")
    if tiny_dir.exists() and not args.rerun_all:
        _print_results(validate_tiny_collection(max_files=args.max_tiny_batches), "Tiny collection")

    steps: List[Tuple[str, List[str], List[Callable[[], List[ValidationItem]]]]] = []

    if args.rerun_all or not khan_path.exists():
        steps.append((
            "collect_khan_academy",
            [sys.executable, "collect_khan_academy.py"],
            [lambda: validate_khan_collection(max_records=args.max_tiny_batches)],
        ))

    if args.rerun_all or not tiny_dir.exists():
        steps.append((
            "collect_tiny_textbooks",
            [sys.executable, "collect_tiny_textbooks.py"],
            [lambda: validate_tiny_collection(max_files=args.max_tiny_batches)],
        ))

    if args.rerun_all or not concept_out.exists():
        steps.append((
            "extract_khan_taxonomy",
            [sys.executable, "extract_khan_taxonomy.py"],
            [validate_taxonomy_outputs],
        ))
    else:
        _print_results(validate_taxonomy_outputs(), "Taxonomy extraction")

    steps.append((
        "build_corpus_index",
        [sys.executable, "build_corpus_index.py"],
        [validate_corpus_index],
    ))
    steps.append((
        "compute_metrics",
        [sys.executable, "compute_metrics.py"],
        [
            lambda: validate_analysis_outputs(max_records=args.max_metrics_records),
            lambda: validate_manifest(require_gates=args.require_gates),
        ],
    ))
    steps.append(("build_dashboard", [sys.executable, "build_dashboard.py"], [validate_dashboard]))

    for label, cmd, checks in steps:
        _run_and_validate(label, cmd, env, checks)

    print("\n[Phase-1] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
