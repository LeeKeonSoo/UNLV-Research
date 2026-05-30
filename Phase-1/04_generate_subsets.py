#!/usr/bin/env python3
"""Canonical entrypoint: generate threshold-controlled subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_eval_common import DEFAULT_PROFILE_CONFIG
from policy.subsets import generate_subsets


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate threshold-controlled subsets.")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_CONFIG)
    args = parser.parse_args()
    manifest = generate_subsets(profiles_path=args.profiles)
    for profile_name, profile_summary in manifest["profiles"].items():
        print(f"[04] profile={profile_name}")
        for dataset, dataset_summary in profile_summary["datasets"].items():
            stage_c = dataset_summary.get("stage_c_core_validation") or {}
            utility_details = dataset_summary.get("utility_probe_details") or {}
            utility_aggregate = utility_details.get("aggregate") or {}
            utility_evidence = utility_aggregate.get("utility_evidence_summary") or {}
            utility_score = dataset_summary.get("small_lm_probe_gain_score", dataset_summary.get("fixed_token_probe_gain_score", 0.0))
            print(
                f"     {dataset}: selected={dataset_summary['selected_records']} "
                f"ratio={dataset_summary['selection_ratio']:.3f} "
                f"coverage={dataset_summary['subset_coverage_retention_score']:.3f} "
                f"utility={utility_score:.6f} "
                f"strict_min={utility_evidence.get('strict_min_gain', 0.0):.6f} "
                f"signal={utility_evidence.get('signal_status', '-')} "
                f"stage_c_pass={bool(stage_c.get('passed'))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
