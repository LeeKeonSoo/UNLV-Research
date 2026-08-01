#!/usr/bin/env python3
"""Regression checks for repository metadata enrichment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _module():
    path = PROJECT_DIR / "65_enrich_temporal_code_repositories.py"
    spec = importlib.util.spec_from_file_location("temporal_code_enrichment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureClient:
    def get_tree_paths(self, repository_identity: str, branch: str):
        if repository_identity == "fixture/clean":
            return ["pyproject.toml", "src/pkg.py", "tests/test_pkg.py"]
        return ["README.md", "src/pkg.py"]

    def merged_pr_metadata(self, repository_identity: str, *, start: str, end: str, limit: int):
        if repository_identity == "fixture/clean":
            return {
                "issue_count": 2,
                "samples": [
                    {
                        "number": 1,
                        "mergedAt": f"{start}T00:00:00Z",
                        "baseRefName": "main",
                        "headRefOid": "1" * 40,
                        "mergeCommit": {"oid": "2" * 40},
                    }
                ],
            }
        return {"issue_count": 0, "samples": []}


def main() -> int:
    module = _module()
    protocol = {
        "collection_contract": {
            "training_window": {"start": "2025-05-01", "end": "2025-12-31"},
            "development_holdout_window": {"start": "2026-01-01", "end": "2026-02-28"},
            "frozen_confirmatory_holdout_window": {"start": "2026-03-01", "end": "2026-05-31"},
        }
    }
    clean = module.enrich_repository(
        {"repository_identity": "fixture/clean", "default_branch": "main"},
        "train",
        protocol,
        FixtureClient(),
        pr_sample_limit=5,
    )
    assert clean["eligible_for_reproducibility_probe"] is True, clean
    assert clean["prose_fields_requested"] is False
    assert clean["code_content_requested"] is False
    assert clean["eligible_for_frozen_repository_manifest"] is False

    weak = module.enrich_repository(
        {"repository_identity": "fixture/weak", "default_branch": "main"},
        "development",
        protocol,
        FixtureClient(),
        pr_sample_limit=5,
    )
    assert weak["eligible_for_reproducibility_probe"] is False
    assert "no_test_suite_path_evidence" in weak["blockers"]
    assert "no_merged_pr_in_assigned_window" in weak["blockers"]
    print("[temporal-code-enrichment] path-only and prose-free enrichment gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
