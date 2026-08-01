#!/usr/bin/env python3
"""Validate frozen binary/log-count/common-random proxy data arms."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "179_freeze_redundancy_saturation_proxy_arms.py"
    spec = importlib.util.spec_from_file_location("redundancy_saturation_proxy_arms", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["chunk_uid"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    source = (
        ROOT
        / "outputs"
        / "temporal_code_collection"
        / "stage_a_code_domain_v2_balanced"
        / "train"
        / "stage_a_pass.jsonl"
    )
    if not source.exists():
        print("[redundancy-proxy-arms] skipped: frozen Stage-A pool unavailable")
        return 0
    module = _load()
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        report = module.freeze(
            source,
            ROOT / "configs" / "temporal_code_curation_protocol_v1.json",
            ROOT / "configs" / "temporal_code_redundancy_saturation_proxy_candidate_v1.json",
            output,
        )
        binary = _ids(output / "binary_current_equal_budget.jsonl")
        log_count = _ids(output / "log_count_equal_budget.jsonl")
        random = _ids(output / "stageA_random_common_disjoint_equal_budget.jsonl")
    assert report["status"] == "redundancy_saturation_proxy_arms_frozen"
    assert not report["blockers"]
    assert binary != log_count
    assert not random.intersection(binary)
    assert not random.intersection(log_count)
    assert report["disjointness"]["common_random_disjoint_from_both_selectors"] is True
    assert all(row["materialized_covers_cap"] for row in report["arms"].values())
    assert report["training_budget"]["target_tokenizer_exact_packing_deferred"] is True
    assert report["utility_scope"].startswith("Stage C only")
    print("[redundancy-proxy-arms] equal cap, selector difference, and common disjoint random: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
