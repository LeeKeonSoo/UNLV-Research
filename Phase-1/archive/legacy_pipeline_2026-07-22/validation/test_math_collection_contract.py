#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    path = ROOT / "collect_math_candidate_pool.py"
    spec = importlib.util.spec_from_file_location("math_raw_mixed_collection", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_module()
    config = module.load_collection_config(ROOT / "configs" / "math_collection_contract.json")
    assert config["raw_pool"]["target_token_proxy"] == 5_000_000
    assert config["reference_pool"]["target_token_proxy"] == 1_000_000
    assert module.passes_lexical_quarantine(
        "A proof of the triangle inequality starts by applying the norm axioms, then bounds each term and concludes the result for all vectors.",
        config,
    )
    assert not module.passes_lexical_quarantine("GSM8K answer key", config)
    assert not module.passes_lexical_quarantine("short", config)
    assert {row["name"] for row in config["benchmark_quarantine"]} == {"GSM8K", "MATH"}
    print("[math-raw-mixed-5m-collection] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
