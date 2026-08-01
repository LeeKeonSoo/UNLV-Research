#!/usr/bin/env python3
"""Operate the frozen forward-development collection pipeline."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from data_eval_common import OUTPUT_DIR


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
SNAPSHOTS = COLLECTION / "forward_development_snapshots"


def refresh() -> None:
    schedule = importlib.import_module("121_freeze_temporal_code_forward_collection_schedule")
    ledger = importlib.import_module("123_build_temporal_code_forward_candidate_ledger")
    status = importlib.import_module("124_build_temporal_code_forward_operations_status")
    schedule.freeze(schedule.DEFAULT_CONTRACT, schedule.DEFAULT_ACCUMULATION, schedule.DEFAULT_OUTPUT)
    ledger.build(ledger.DEFAULT_SCHEDULE, sorted(SNAPSHOTS.glob("*.json")), ledger.DEFAULT_OUTPUT)
    status.build(status.DEFAULT_SCHEDULE, status.DEFAULT_LEDGER, status.DEFAULT_OUTPUT)


def collect(shard_index: int, available_through: str, delay_seconds: float) -> None:
    refresh()
    discovery_module = importlib.import_module("64_discover_temporal_code_repositories")
    helper = importlib.import_module("110_discover_temporal_code_forward_e2_pilot")
    collector = importlib.import_module("122_collect_temporal_code_forward_snapshot_shard")
    token, _ = discovery_module.resolve_github_token()
    if not token:
        raise RuntimeError("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    output = SNAPSHOTS / f"{available_through}__shard_{shard_index:03d}.json"
    if output.exists():
        raise RuntimeError(f"Immutable snapshot already exists: {output}")
    collector.collect(
        collector.DEFAULT_SCHEDULE,
        collector.DEFAULT_DISCOVERY,
        output,
        shard_index,
        available_through,
        helper.Client(token, max(0.0, delay_seconds)),
    )
    refresh()


def main() -> int:
    parser = argparse.ArgumentParser(description="Operate frozen forward development collection.")
    parser.add_argument("--action", choices=("refresh", "collect"), default="refresh")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--available-through")
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if args.action == "collect":
        if args.shard_index is None or not args.available_through:
            raise SystemExit("--shard-index and --available-through are required for collect.")
        collect(args.shard_index, args.available_through, args.delay_seconds)
    else:
        refresh()
    status_path = OUTPUT_DIR / "validation" / "temporal_code_forward_operations_status.json"
    print(status_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
