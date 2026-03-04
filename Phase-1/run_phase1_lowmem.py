#!/usr/bin/env python3
"""Run Phase-1 pipeline with final-validation defaults and memory-safe settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from phase1_autosave import autosave_stage, resolve_autosave_target
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


def _load_dataset_names(config_path: str) -> List[str]:
    cfg = Path(config_path)
    if not cfg.is_absolute():
        cfg = PROJECT_DIR / cfg
    if not cfg.exists():
        return ["khan_academy", "tiny_textbooks"]
    try:
        with cfg.open("r", encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
    except Exception:
        return ["khan_academy", "tiny_textbooks"]
    raw_specs = payload.get("datasets", []) if isinstance(payload, dict) else payload
    names: List[str] = []
    if isinstance(raw_specs, list):
        for i, spec in enumerate(raw_specs):
            if not isinstance(spec, dict):
                names.append(f"dataset_{i+1}")
                continue
            name = str(spec.get("name") or f"dataset_{i+1}").strip().lower()
            names.append(name)
    return names or ["khan_academy", "tiny_textbooks"]


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
    parser = argparse.ArgumentParser(
        description="Run Phase-1 pipeline with final-validation defaults and memory-safe settings"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced-batch smoke run (not for final reporting).",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing collection/taxonomy inputs when present.",
    )
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
        "--allow-gate-fail",
        action="store_true",
        help="Compatibility flag. Prefer --require-gates for strict pre-check.",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="Fail compute-step validation when any required gate is not passed.",
    )
    parser.add_argument(
        "--datasets-config",
        default="datasets_config.json",
        help="Dataset config path (relative to Phase-1 root).",
    )
    parser.add_argument(
        "--identity-config",
        default="configs/metric_identity_v1.json",
        help="Metric identity config path (relative to Phase-1 root).",
    )
    parser.add_argument(
        "--autosave-root",
        default="",
        help=(
            "Autosave destination root. "
            "If omitted, auto-detects Google Drive in Colab. "
            "Set to 'off' to disable."
        ),
    )
    parser.add_argument(
        "--disable-autosave",
        action="store_true",
        help="Disable autosave sync after each step.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    rerun_all = not args.reuse_existing
    require_gates = bool(args.require_gates) and not bool(args.allow_gate_fail)

    # Final-validation defaults with RAM-safe redundancy settings.
    profile: Dict[str, str] = {
        "PHASE1_DEVICE": "auto",
        "PHASE1_CUDA_DEVICE": "0",
        "PHASE1_USE_GPU": "1",
        "PHASE1_DOMAIN_BATCH_SIZE": "256",
        "PHASE1_MAX_BATCHES": "0",
        "PHASE1_INDEX_MAX_BATCHES": "0",
        "PHASE1_TFIDF_MAX_FEATURES": "3000",
        "PHASE1_BUILD_TFIDF_MATRIX": "0",
        "PHASE1_STORE_DOC_TEXTS": "1",
        "PHASE1_DOC_TEXT_BACKEND": "sqlite",
        "PHASE1_DOC_TEXT_INSERT_BATCH": "2000",
        "PHASE1_DOC_TEXT_CACHE_LIMIT": "2500",
        "PHASE1_ENABLE_MINHASH": "1",
        "PHASE1_QUERY_CACHE_LIMIT": "1024",
        "PHASE1_NGRAM_CACHE_LIMIT": "2048",
        "PHASE1_SEMANTIC_CANDIDATE_LIMIT": "80",
        "PHASE1_NGRAM_CANDIDATE_LIMIT": "80",
        "PHASE1_SKIP_REDUNDANCY": "0",
        "PHASE1_SKIP_PERPLEXITY": "0",
    }

    if args.quick:
        profile["PHASE1_MAX_BATCHES"] = "1"
        profile["PHASE1_INDEX_MAX_BATCHES"] = "1"
        profile["PHASE1_SKIP_REDUNDANCY"] = "1"
        profile["PHASE1_ENABLE_MINHASH"] = "0"
        profile["PHASE1_STORE_DOC_TEXTS"] = "0"
        profile["PHASE1_DOC_TEXT_BACKEND"] = "memory"

    env = os.environ.copy()
    env.update(profile)
    env["PHASE1_DATASETS_CONFIG"] = args.datasets_config
    env["PHASE1_IDENTITY_CONFIG"] = args.identity_config

    autosave_target: Optional[Path] = None
    autosave_root = (args.autosave_root or "").strip() or None
    if args.disable_autosave:
        print("[Autosave] disabled by --disable-autosave")
    else:
        autosave_target = resolve_autosave_target(PROJECT_DIR, autosave_root)
        if autosave_target is None:
            print("[Autosave] disabled by config/env")
        elif autosave_target.resolve() == PROJECT_DIR.resolve():
            print(f"[Autosave] local mode (project path): {PROJECT_DIR}")
        else:
            print(f"[Autosave] external target: {autosave_target}")

    print("[Phase-1] Using execution profile:")
    for k in sorted(profile):
        print(f"  {k}={env[k]}")
    print(f"  PHASE1_DATASETS_CONFIG={env['PHASE1_DATASETS_CONFIG']}")
    print(f"  PHASE1_IDENTITY_CONFIG={env['PHASE1_IDENTITY_CONFIG']}")

    dataset_names = set(_load_dataset_names(args.datasets_config))

    khan_path = PROJECT_DIR / "khan_k12_concepts" / "all_k12_concepts.json"
    tiny_dir = PROJECT_DIR / "tiny_textbooks_raw"
    outputs = PROJECT_DIR / "outputs"
    concept_out = outputs / "concept_prototypes_tfidf.pkl"

    # Always validate pre-existing inputs before deciding which stages to skip.
    if "khan_academy" in dataset_names and khan_path.exists() and not rerun_all:
        _print_results(validate_khan_collection(max_records=args.max_tiny_batches), "Khan collection")
    if "tiny_textbooks" in dataset_names and tiny_dir.exists() and not rerun_all:
        _print_results(validate_tiny_collection(max_files=args.max_tiny_batches), "Tiny collection")

    steps: List[Tuple[str, List[str], List[Callable[[], List[ValidationItem]]]]] = []

    if "khan_academy" in dataset_names and (rerun_all or not khan_path.exists()):
        steps.append((
            "collect_khan_academy",
            [sys.executable, "01_collect_khan_academy.py"],
            [lambda: validate_khan_collection(max_records=args.max_tiny_batches)],
        ))

    if "tiny_textbooks" in dataset_names and (rerun_all or not tiny_dir.exists()):
        steps.append((
            "collect_tiny_textbooks",
            [sys.executable, "02_collect_tiny_textbooks.py"],
            [lambda: validate_tiny_collection(max_files=args.max_tiny_batches)],
        ))

    if rerun_all or not concept_out.exists():
        steps.append((
            "extract_khan_taxonomy",
            [sys.executable, "03_extract_khan_taxonomy.py"],
            [validate_taxonomy_outputs],
        ))
    else:
        _print_results(validate_taxonomy_outputs(), "Taxonomy extraction")

    steps.append((
        "build_corpus_index",
        [sys.executable, "04_build_corpus_index.py"],
        [validate_corpus_index],
    ))
    steps.append((
        "compute_metrics",
        [sys.executable, "05_compute_metrics.py"],
        [
            lambda: validate_analysis_outputs(max_records=args.max_metrics_records),
            lambda: validate_manifest(require_gates=require_gates),
        ],
    ))
    steps.append(("build_dashboard", [sys.executable, "06_build_dashboard.py"], [validate_dashboard]))

    for label, cmd, checks in steps:
        _run_and_validate(label, cmd, env, checks)
        if not args.disable_autosave:
            autosave_stage(
                project_dir=PROJECT_DIR,
                stage=label,
                autosave_root=autosave_root,
                resolved_target=autosave_target,
            )

    if not args.disable_autosave:
        autosave_stage(
            project_dir=PROJECT_DIR,
            stage="final",
            autosave_root=autosave_root,
            resolved_target=autosave_target,
        )

    print("\n[Phase-1] Pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
