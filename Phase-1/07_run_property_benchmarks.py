#!/usr/bin/env python3
"""Run property-based metric benchmarks against scored outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from property_benchmarks import (
    DEFAULT_OUT_DIR,
    DEFAULT_SCORED_DIR,
    benchmark_scored_dataset,
    write_benchmark_report,
)


def _resolve_datasets(dataset_arg: str, scored_dir: Path) -> List[str]:
    raw = (dataset_arg or "").strip()
    if raw and raw not in {"all", "auto"}:
        return [raw]
    datasets: List[str] = []
    for path in sorted(scored_dir.glob("*.jsonl")):
        datasets.append(path.stem)
    return datasets


def main() -> int:
    parser = argparse.ArgumentParser(description="Run property-based benchmarks for scored datasets.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--scored-dir", type=Path, default=DEFAULT_SCORED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--min-assertion-bucket-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = _resolve_datasets(args.dataset, args.scored_dir)
    if not datasets:
        raise SystemExit(f"No scored datasets found in: {args.scored_dir}")

    total_supported = 0
    total_passed = 0
    total_failed = 0
    for dataset in datasets:
        scored_path = args.scored_dir / f"{dataset}.jsonl"
        if not scored_path.exists():
            raise SystemExit(f"Scored dataset not found: {scored_path}")
        report = benchmark_scored_dataset(
            scored_path=scored_path,
            dataset_name=dataset,
            sample_limit=args.sample_limit,
            min_assertion_bucket_size=args.min_assertion_bucket_size,
            seed=args.seed,
        )
        out_path = write_benchmark_report(report, out_dir=args.out_dir)
        summary = report["summary"]
        total_supported += int(summary["supported_assertions"])
        total_passed += int(summary["passed_assertions"])
        total_failed += int(summary["failed_assertions"])
        print(f"[07] property benchmark: {out_path}")
        print(
            f"[07] {dataset}: assertions supported={summary['supported_assertions']} "
            f"passed={summary['passed_assertions']} failed={summary['failed_assertions']}"
        )
        print(f"[07] {dataset}: near-dup saturation: {report['near_duplicate_distribution']}")
    print(
        f"[07] aggregate: assertions supported={total_supported} "
        f"passed={total_passed} failed={total_failed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
