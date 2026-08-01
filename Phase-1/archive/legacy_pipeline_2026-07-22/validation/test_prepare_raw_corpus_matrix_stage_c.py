#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    path = ROOT / "234_prepare_raw_corpus_matrix_stage_c.py"
    spec = importlib.util.spec_from_file_location("prepare_raw_corpus_matrix_stage_c", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load()
    with tempfile.TemporaryDirectory() as temporary_dir:
        report = module.prepare(
            ROOT / "configs" / "raw_corpus_matrix_stage_c_study_v1.json",
            Path(temporary_dir),
            allow_download=False,
        )
    assert report["status"] == "raw_corpus_matrix_stage_c_inputs_frozen"
    assert report["equal_token_arms"]["curated_equal_token"]["target_tokens_with_eos"] > 0
    packed = report["token_blocks"]["equal_token_blocks"]
    assert packed["curated_equal_token"]["packed_tokens"] == packed["stage_a_random_equal_token"]["packed_tokens"]
    assert packed["curated_equal_token"]["packed_tokens"] == packed["raw_mixed_random_equal_token"]["packed_tokens"]
    assert report["equal_token_arms"]["curated_common_baseline_overlap_count"] == 0
    assert report["common_stage_a_baseline_sha256"]
    assert report["natural_budget_arms"]["raw_mixed_all_natural"]["target_tokens_with_eos"] > report["natural_budget_arms"]["curated_natural"]["target_tokens_with_eos"]
    assert report["holdouts"]["development"]["stage_b_read"] is False
    assert report["holdouts"]["confirmatory"]["stage_b_read"] is False
    assert report["holdouts"]["confirmatory"]["stage_a_pass_count"] > 0
    print("[prepare-raw-corpus-matrix-stage-c] equal-token arms and Stage-A-only holdouts: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
