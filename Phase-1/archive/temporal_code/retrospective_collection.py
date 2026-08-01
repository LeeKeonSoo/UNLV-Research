#!/usr/bin/env python3
"""Collect all missing retrospective-development metadata shards."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from data_eval_common import OUTPUT_DIR, load_json


COLLECTION = OUTPUT_DIR / "temporal_code_collection"
DEFAULT_SCHEDULE = COLLECTION / "temporal_code_retrospective_development_schedule.json"
DEFAULT_DISCOVERY = COLLECTION / "forward_development_repository_discovery_combined.json"
DEFAULT_OUTPUT_DIR = COLLECTION / "retrospective_development_snapshots"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect all missing retrospective development shards.")
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay-seconds", type=float, default=0.1)
    args = parser.parse_args()
    schedule = load_json(args.schedule)
    discovery_module = importlib.import_module("64_discover_temporal_code_repositories")
    helper = importlib.import_module("110_discover_temporal_code_forward_e2_pilot")
    collector = importlib.import_module("128_collect_temporal_code_retrospective_shard")
    token, _ = discovery_module.resolve_github_token()
    if not token:
        raise SystemExit("Authenticated GitHub CLI or GITHUB_TOKEN is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for shard in schedule["shards"]:
        output = args.output_dir / f"retrospective__shard_{shard['shard_index']:03d}.json"
        if output.exists():
            skipped += 1
            continue
        report = collector.collect(
            args.schedule,
            args.discovery,
            output,
            int(shard["shard_index"]),
            helper.Client(token, max(0.0, args.delay_seconds)),
        )
        completed += 1
        print({"shard": shard["shard_id"], "summary": report["summary"]})
    print({"completed_shards": completed, "skipped_existing_shards": skipped})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
