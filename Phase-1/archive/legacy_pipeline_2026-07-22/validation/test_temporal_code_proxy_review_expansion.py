#!/usr/bin/env python3
"""Regression checks for train-only temporal-code proxy-review expansion plans."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data_eval_common import load_json  # noqa: E402


def main() -> int:
    module = importlib.import_module("80_freeze_temporal_code_proxy_review_expansion")
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json")
    discovery = {
        "candidates": {
            "fixture/used": {"repository_url": "https://github.com/fixture/used", "license": "MIT"},
            "fixture/small": {"repository_url": "https://github.com/fixture/small", "license": "MIT"},
            "fixture/large": {"repository_url": "https://github.com/fixture/large", "license": "Apache-2.0"},
            "fixture/dev": {"repository_url": "https://github.com/fixture/dev", "license": "MIT"},
        }
    }
    enrichment = {
        "repositories": {
            "fixture/used": {"assigned_split": "train", "tree_evidence": {"tree_path_count": 10}, "merged_pr_evidence": {"issue_count": 5, "samples": []}},
            "fixture/small": {"assigned_split": "train", "tree_evidence": {"tree_path_count": 20}, "merged_pr_evidence": {"issue_count": 5, "samples": []}},
            "fixture/large": {"assigned_split": "train", "tree_evidence": {"tree_path_count": 30}, "merged_pr_evidence": {"issue_count": 5, "samples": []}},
            "fixture/dev": {"assigned_split": "development", "tree_evidence": {"tree_path_count": 1}, "merged_pr_evidence": {"issue_count": 5, "samples": []}},
        }
    }
    reproducibility = {
        "repositories": {
            identity: {"eligible_for_quarantine_review": True}
            for identity in discovery["candidates"]
        }
    }
    smoke = {"selected_repositories": {"train": {"repository_identity": "fixture/used"}}}
    plan = module.freeze(protocol, discovery, enrichment, reproducibility, smoke)
    assert list(plan["selected_repositories"]) == ["train"], plan
    assert plan["selected_repositories"]["train"]["repository_identity"] == "fixture/small", plan
    assert plan["review_scope"]["training_approval"] is False, plan
    assert plan["review_scope"]["development_or_confirmatory_content"] == "forbidden", plan
    next_plan = module.freeze(
        protocol,
        discovery,
        enrichment,
        reproducibility,
        smoke,
        [{"selected_repositories": {"train": {"repository_identity": "fixture/small"}}}],
    )
    assert next_plan["selected_repositories"]["train"]["repository_identity"] == "fixture/large", next_plan
    assert next_plan["excluded_repository_identities"] == ["fixture/small", "fixture/used"], next_plan
    print("[temporal-code-proxy-review-expansion] train-only second repository plan: pass")
    print("[temporal-code-proxy-review-expansion] prior expansion exclusion: pass")
    print("[temporal-code-proxy-review-expansion] training approval remains forbidden: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
