#!/usr/bin/env python3
"""Regression checks for metadata-only repository discovery."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import importlib.util  # noqa: E402

from data_eval_common import load_json  # noqa: E402


def _module():
    path = PROJECT_DIR / "64_discover_temporal_code_repositories.py"
    spec = importlib.util.spec_from_file_location("temporal_code_discovery", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _module()
    protocol = load_json(PROJECT_DIR / "configs" / "temporal_code_curation_protocol_v1.json")
    queries = module.build_search_queries(protocol, min_stars=20)
    assert len(queries) == len(protocol["collection_contract"]["allowed_licenses"])
    assert all("language:Python" in query and "fork:false" in query and "archived:false" in query for query in queries)
    token, source = module.resolve_github_token()
    assert source in {"environment", "github_cli", "none"}
    assert token is None or isinstance(token, str)

    benchmark_patterns = {"openai/human-eval"}
    clean = module.repository_candidate(
        {
            "full_name": "Example/Clean",
            "html_url": "https://github.com/Example/Clean",
            "url": "https://api.github.com/repos/Example/Clean",
            "license": {"spdx_id": "MIT"},
            "fork": False,
            "archived": False,
            "stargazers_count": 10,
        },
        benchmark_patterns,
    )
    assert clean["metadata_only"] is True
    assert clean["eligible_for_metadata_enrichment"] is True
    assert clean["eligible_for_frozen_repository_manifest"] is False
    assert clean["discovery_status"] == "metadata_discovered_pending_enrichment"

    benchmark = module.repository_candidate(
        {
            "full_name": "openai/human-eval",
            "license": {"spdx_id": "MIT"},
            "fork": False,
            "archived": False,
        },
        benchmark_patterns,
    )
    assert benchmark["eligible_for_metadata_enrichment"] is False
    assert "benchmark_repository" in benchmark["blockers"]
    print("[temporal-code-discovery] metadata-only queries and repository gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
