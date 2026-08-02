#!/usr/bin/env python3
# Run: python scripts/build_math_general_benchmark_snapshots.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from benchmark_snapshot import build_benchmark_registry
from benchmark_snapshot_contract import load_benchmark_snapshot_registry


DEFAULT_REGISTRY = ROOT / "protocols" / "math_general_benchmark_snapshot_registry_v1.json"
DEFAULT_CACHE = Path("D:/UNLV-Research/hf_cache/hub")
DEFAULT_OUTPUT = Path("D:/UNLV-Research/benchmark_snapshots_v1/math_general")
DEFAULT_MANIFEST = ROOT / "validation" / "frozen_contracts" / "math_general_benchmark_snapshot_manifest_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical Math and General benchmark contamination snapshots.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    registry = load_benchmark_snapshot_registry(args.registry)
    frozen = build_benchmark_registry(registry, args.cache_root, args.output_root)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(frozen.model_dump_json(indent=2) + "\n", encoding="utf-8")
    counts = ", ".join(f"{item.benchmark_id}={item.task_count}" for item in frozen.snapshots)
    print(f"[math-general-benchmark-snapshots] {counts} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
