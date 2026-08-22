#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_baselines.nemo_curator_code import load_frozen_recipe


def main() -> int:
    recipe = load_frozen_recipe(ROOT / "protocols" / "code_7m_nemo_curator_baseline_v1.json")
    assert recipe.image_digest == "sha256:96abe1a74557d0cd20a6288f5b38804d0dba6c220a7c1e221607e32fe13b5710"
    assert recipe.input_sha256 == "804dc90e35b360ae257fba99cdb1835d4b72ebd174528650dcdd20d9621a58e7"
    assert recipe.target_token_budget is None
    assert recipe.code_filters == (
        "PythonCommentToCodeFilter",
        "NumberOfLinesOfCodeFilter",
        "AlphaFilter",
        "XMLHeaderFilter",
    )
    assert recipe.fuzzy_seed == 42
    assert recipe.fuzzy_char_ngrams == 24
    assert recipe.fuzzy_num_bands == 20
    assert recipe.fuzzy_minhashes_per_band == 13
    print("[nemo-curator-baseline] frozen recipe: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
