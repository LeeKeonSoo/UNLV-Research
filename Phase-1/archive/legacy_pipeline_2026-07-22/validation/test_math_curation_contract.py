#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module():
    path = ROOT / "curate_math_candidate_pool.py"
    spec = importlib.util.spec_from_file_location("math_raw_mixed_abc_curation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_module()
    config = module.load_config(ROOT / "configs" / "math_curation_contract.json")
    source = {
        "record_uid": "fixture-1",
        "text": "A sufficiently long mathematical derivation explains each step and proves the stated result using explicit assumptions.",
        "source_dataset_id": "AI-MO/NuminaMath-CoT",
        "source_split": "train",
        "source_row_index": 1,
        "pool_role": "known_high_quality_reference_context",
        "token_proxy": 16,
    }
    candidate = module.stage_a_candidate(source, config, "2026-07-22")
    assert candidate["rights"]["status"] == "allowed"
    assert candidate["rights"]["license"] == "Apache-2.0"
    assert candidate["partition"]["source_pool_role"] == "known_high_quality_reference_context"
    chunks = module.chunk_text(candidate["text"], 48)
    assert len(chunks) >= 2
    assert all(chunk.strip() for chunk in chunks)
    assert sum(len(chunk.split()) for chunk in chunks) == len(candidate["text"].split())
    short_text = "A short proof has one premise and one conclusion."
    short_chunks = module.chunk_text(short_text, 80)
    assert short_chunks == [short_text]
    print("[math-raw-mixed-abc-curation] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
