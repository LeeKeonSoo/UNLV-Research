#!/usr/bin/env python3
"""Regression checks for commit-identity reproducibility probes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _module():
    path = PROJECT_DIR / "66_probe_temporal_code_commit_reproducibility.py"
    spec = importlib.util.spec_from_file_location("temporal_code_reproducibility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixtureClient:
    def __init__(self, parent: bool = True):
        self.parent = parent

    def commit_identities(self, repository_identity: str, shas):
        return {
            sha: {
                "oid": sha,
                "parents": {"nodes": [{"oid": "a" * 40}] if self.parent else []},
            }
            for sha in shas
        }


def _row():
    return {
        "repository_identity": "fixture/clean",
        "assigned_split": "train",
        "merged_pr_evidence": {
            "samples": [{"mergeCommit": {"oid": "b" * 40}}],
        },
    }


def main() -> int:
    module = _module()
    clean = module.probe_repository(_row(), FixtureClient())
    assert clean["eligible_for_quarantine_review"] is True, clean
    assert clean["eligible_for_frozen_repository_manifest"] is False
    assert clean["prose_fields_requested"] is False
    assert clean["code_content_requested"] is False
    missing_parent = module.probe_repository(_row(), FixtureClient(parent=False))
    assert "sampled_parent_commit_not_fetchable" in missing_parent["blockers"]
    print("[temporal-code-reproducibility] commit identity and parent gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
