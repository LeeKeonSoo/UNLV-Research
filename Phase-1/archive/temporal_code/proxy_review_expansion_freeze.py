#!/usr/bin/env python3
"""Freeze a train-only additional-repository plan for blind Stage-B proxy review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from data_eval_common import OUTPUT_DIR, load_json, save_json


DEFAULT_PROTOCOL = Path("configs") / "temporal_code_curation_protocol_v1.json"
DEFAULT_DISCOVERY = OUTPUT_DIR / "temporal_code_collection" / "repository_candidate_manifest_authenticated.json"
DEFAULT_ENRICHMENT = OUTPUT_DIR / "temporal_code_collection" / "repository_enrichment_report_smoke30.json"
DEFAULT_REPRODUCIBILITY = OUTPUT_DIR / "temporal_code_collection" / "commit_reproducibility_report_smoke30.json"
DEFAULT_SMOKE_PLAN = OUTPUT_DIR / "temporal_code_collection" / "temporal_code_smoke_fetch_plan.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "temporal_code_collection" / "proxy_review_expansion_fetch_plan.json"


def freeze(
    protocol: Dict[str, Any],
    discovery: Dict[str, Any],
    enrichment: Dict[str, Any],
    reproducibility: Dict[str, Any],
    smoke_plan: Dict[str, Any],
    exclusion_plans: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    allowed = set(protocol["collection_contract"]["allowed_licenses"])
    excluded = {
        row["repository_identity"]
        for row in (smoke_plan.get("selected_repositories") or {}).values()
        if isinstance(row, dict)
    }
    for plan in exclusion_plans:
        excluded.update(
            row["repository_identity"]
            for row in (plan.get("selected_repositories") or {}).values()
            if isinstance(row, dict)
        )
    candidates = []
    for identity, row in enrichment["repositories"].items():
        if row["assigned_split"] != "train" or identity in excluded:
            continue
        reproducible = (reproducibility.get("repositories") or {}).get(identity) or {}
        discovery_row = discovery["candidates"][identity]
        if reproducible.get("eligible_for_quarantine_review") is not True:
            continue
        if discovery_row.get("license") not in allowed:
            continue
        candidates.append(
            {
                "repository_identity": identity,
                "repository_url": discovery_row["repository_url"],
                "license": discovery_row["license"],
                "assigned_split": "train",
                "tree_path_count": row["tree_evidence"]["tree_path_count"],
                "merged_pr_count_in_window": row["merged_pr_evidence"]["issue_count"],
                "sampled_prs": row["merged_pr_evidence"]["samples"],
            }
        )
    candidates.sort(key=lambda row: (int(row["tree_path_count"]), row["repository_identity"]))
    if not candidates:
        raise RuntimeError("No train-only proxy-review expansion candidate is available.")
    selected = candidates[0]
    return {
        "schema_version": "temporal-code-proxy-review-expansion-plan-v1",
        "status": "frozen_before_content_fetch",
        "selection_rule": (
            "Choose the smallest path-count train-split repository not used by the first smoke "
            "or an explicitly excluded prior expansion, "
            "with allowlisted license and reproducible sampled commit identities."
        ),
        "excluded_repository_identities": sorted(excluded),
        "selected_repositories": {"train": selected},
        "content_fetch_limits": {
            "maximum_pull_requests_per_repository": 2,
            "maximum_changed_files_per_pull_request": 50,
            "maximum_file_bytes": 524288,
            "allowed_file_suffixes": [".py", ".md", ".rst", ".toml", ".cfg", ".ini", ".txt"],
            "issue_and_pull_request_prose": "do_not_fetch_for_training_payload",
            "binary_generated_vendor_lock_files": "exclude",
        },
        "review_scope": {
            "purpose": "additional-repository blind Stage-B Core-proxy review",
            "stage0_release_candidate": False,
            "training_approval": False,
            "development_or_confirmatory_content": "forbidden",
        },
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Proxy-review expansion plan only; fetched content cannot enter training or Stage C.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze temporal-code proxy-review expansion.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--enrichment", type=Path, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--reproducibility", type=Path, default=DEFAULT_REPRODUCIBILITY)
    parser.add_argument("--smoke-plan", type=Path, default=DEFAULT_SMOKE_PLAN)
    parser.add_argument(
        "--exclude-plan",
        type=Path,
        action="append",
        default=[],
        help="Prior expansion plan whose selected repositories must be excluded; repeatable.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plan = freeze(
        load_json(args.protocol),
        load_json(args.discovery),
        load_json(args.enrichment),
        load_json(args.reproducibility),
        load_json(args.smoke_plan),
        [load_json(path) for path in args.exclude_plan],
    )
    save_json(args.output, plan)
    print(plan["selected_repositories"]["train"]["repository_identity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
