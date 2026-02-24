#!/usr/bin/env python3
"""Run Phase-1 pipeline with a memory-sensitive profile.

Defaults are tuned for high RAM pressure while preserving normal metric outputs:
- corpus index built on first 1 tiny batch
- reduced vocab TF-IDF
- redundancy disabled for memory stability
- perplexity enabled
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parent


def _run(cmd: List[str], env: Dict[str, str], label: str) -> None:
    """Run a command and fail fast with a readable error."""
    print(f"\n[Phase-1] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR, env=env)
    if result.returncode != 0:
        raise SystemExit(f"[Phase-1] failed at: {label} (code {result.returncode})")


def _safe_exists(p: Path) -> bool:
    return p.exists()


def main() -> int:
    # Tuned low-memory profile defaults (can be overridden by pre-set env vars).
    profile: Dict[str, str] = {
        # Device / batch behavior
        "PHASE1_DEVICE": "auto",
        "PHASE1_CUDA_DEVICE": "0",
        "PHASE1_USE_GPU": "1",
        "PHASE1_DOMAIN_BATCH_SIZE": "256",
        "PHASE1_MAX_BATCHES": "1",               # tune to all for full run

        # Index build profile
        "PHASE1_INDEX_MAX_BATCHES": "1",         # tune to all for full run
        "PHASE1_TFIDF_MAX_FEATURES": "2000",      # keep quality with lower memory
        "PHASE1_BUILD_TFIDF_MATRIX": "1",
        "PHASE1_STORE_DOC_TEXTS": "0",
        "PHASE1_ENABLE_MINHASH": "0",

        # Per-request throttling (cache/candidate controls)
        "PHASE1_QUERY_CACHE_LIMIT": "256",
        "PHASE1_NGRAM_CACHE_LIMIT": "512",
        "PHASE1_SEMANTIC_CANDIDATE_LIMIT": "80",
        "PHASE1_NGRAM_CANDIDATE_LIMIT": "40",

        # Keep metrics enabled (do not skip for quality pass)
        "PHASE1_SKIP_REDUNDANCY": "1",
        "PHASE1_SKIP_PERPLEXITY": "0",
    }

    env = os.environ.copy()
    for k, v in profile.items():
        env.setdefault(k, v)

    print("[Phase-1] Using low-memory profile:")
    for k in sorted(profile):
        print(f"  {k}={env[k]}")

    # Optional data collection: run only if source files are missing.
    khan_path = PROJECT_DIR / "khan_k12_concepts" / "all_k12_concepts.json"
    tiny_dir = PROJECT_DIR / "tiny_textbooks_raw"
    outputs = PROJECT_DIR / "outputs"
    concept_out = outputs / "concept_prototypes_tfidf.pkl"

    commands: List[Tuple[str, List[str]]] = []

    if not khan_path.exists():
        commands.append(("collect_khan_academy", [sys.executable, "collect_khan_academy.py"]))
    if not tiny_dir.exists():
        commands.append(("collect_tiny_textbooks", [sys.executable, "collect_tiny_textbooks.py"]))

    if not concept_out.exists():
        commands.append(("extract_khan_taxonomy", [sys.executable, "extract_khan_taxonomy.py"]))

    commands.append(("build_corpus_index", [sys.executable, "build_corpus_index.py"]))
    commands.append(("compute_metrics", [sys.executable, "compute_metrics.py"]))
    commands.append(("build_dashboard", [sys.executable, "build_dashboard.py"]))

    for label, cmd in commands:
        _run(cmd, env, label)

    print("\n[Phase-1] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
