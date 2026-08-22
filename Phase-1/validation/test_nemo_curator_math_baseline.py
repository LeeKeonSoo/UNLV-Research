from __future__ import annotations

import json
from pathlib import Path

from external_baselines.nemo_curator_math import load_filter_config, load_frozen_recipe


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "math_7m_nemo_curator_baseline_v1.json"


def test_math_recipe_is_natural_budget_and_notation_safe() -> None:
    recipe = load_frozen_recipe(PROTOCOL)
    raw = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert recipe.target_token_budget is None
    assert recipe.input_records == 3231
    assert raw["recipe"]["semantic_dedup"] is False
    assert set(raw["recipe"]["excluded_math_hostile_filters"]).isdisjoint(recipe.code_filters)


def test_math_recipe_freezes_document_repetition_thresholds() -> None:
    config = load_filter_config(PROTOCOL)

    assert config.min_words == 20
    assert config.max_repeated_line_fraction == 0.7
    assert config.max_repeated_paragraphs_ratio == 0.7
    assert config.top_ngram_ratios == ((2, 0.2), (3, 0.18), (4, 0.16))
