#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_policy_module():
    path = ROOT / "policy" / "subsets.py"
    spec = importlib.util.spec_from_file_location("policy_subsets", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_components_expose_selection_value_alias() -> None:
    module = _load_policy_module()
    record = {
        "word_count": 96,
        "core_metrics": {
            "reference_quality_score": {
                "score": 0.82,
                "details": {
                    "lexical_diversity": 0.38,
                    "boilerplate_hits": 0,
                    "word_count": 96,
                },
            },
            "shingle_near_duplicate_risk_score": {
                "score": 0.12,
                "details": {
                    "useful_recurrence_score": 0.35,
                    "intra_chunk_repeat_pressure": 0.24,
                },
            },
            "structural_validity_gate": {"score": 1.0},
        },
        "diagnostic_metrics": {
            "predictive_utility_proxy": {"score": 0.99},
            "explanatory_quality_proxy": {"score": 0.0},
            "tail_cluster_rarity_proxy": {"score": 0.0},
        },
    }

    components = module._objective_components(record)
    axes = module._axis_scores(record)

    assert components["selection_value"] == components["quality"]
    assert components["selection_value_learnability_support"] == components["quality_learnability_support"]
    assert "predictive_utility_proxy" not in components
    assert "diagnostic_predictive_utility" not in axes


def test_stage_b_retain_all_when_no_budget_is_declared() -> None:
    module = _load_policy_module()

    budget = module._resolve_stage_b_budget({}, total_word_count=100)

    assert budget.binding is False
    assert budget.mode == "retain_all"
    assert budget.word_limit is None


def test_stage_b_word_budget_is_binding_and_enforced() -> None:
    module = _load_policy_module()
    profile = {"stage_b_budget": {"max_word_count": 50}}

    budget = module._resolve_stage_b_budget(profile, total_word_count=100)
    fitted = module.fit_word_budget(
        [{"word_count": 30}, {"word_count": 25}, {"word_count": 20}],
        word_count=lambda record: int(record["word_count"]),
        word_limit=int(budget.word_limit or 0),
    )

    assert budget.binding is True
    assert budget.mode == "word_budget"
    assert sum(int(record["word_count"]) for record in fitted) <= 50
    assert fitted == [{"word_count": 30}, {"word_count": 20}]


def test_scorer_source_exposes_grouped_metric_api() -> None:
    source = (ROOT / "signals" / "core.py").read_text(encoding="utf-8")

    assert "def score_chunk_grouped" in source
    assert "\"core_metrics\"" in source
    assert "\"diagnostic_metrics\"" in source


def test_selection_value_weight_works_without_legacy_quality_key() -> None:
    module = _load_policy_module()

    score = module._objective_score_with_constraints(
        record={
            "diagnostics": {"cluster_id": 7, "cluster_size": 1},
            "provenance": {"source": "fixture"},
        },
        components={
            "selection_value": 0.8,
            "redundancy_risk": 0.1,
            "useful_length_support": 0.0,
            "lexical_diversity": 0.0,
            "useful_recurrence": 0.0,
            "learnability_support": 0.0,
            "pattern_recurrence_support": 0.0,
            "quality_tail_penalty": 0.0,
            "boilerplate_penalty": 0.0,
        },
        selector_cfg={
            "objective_weights": {"selection_value": 1.0, "redundancy_risk": 0.5},
            "constraint_penalties": {"rare_cluster_bonus": 0.0, "small_cluster_bonus": 0.0},
        },
        strategy={"rare_clusters": set()},
    )

    assert score == 0.75


def main() -> int:
    test_policy_components_expose_selection_value_alias()
    test_scorer_source_exposes_grouped_metric_api()
    test_selection_value_weight_works_without_legacy_quality_key()
    test_stage_b_retain_all_when_no_budget_is_declared()
    test_stage_b_word_budget_is_binding_and_enforced()
    print("[core-policy-surface] Selection Value and diagnostic contracts: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
