from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luna_normal_and_hard_share_ranker_without_runtime_oracle() -> None:
    configs = {
        mode: json.loads(
            (ROOT / f"configs/code_7m_luna_final_{mode}_v1.json").read_text(
                encoding="utf-8"
            )
        )
        for mode in ("normal", "hard")
    }

    assert {config["curation_mode"] for config in configs.values()} == {
        "normal",
        "hard",
    }
    quality = [config["quality_runtime"] for config in configs.values()]
    assert {item["method"] for item in quality} == {"distilled_quality_ranker_v1"}
    assert len({item["embedding_manifest_path"] for item in quality}) == 1
    assert len({item["ranker_manifest_path"] for item in quality}) == 1
    assert all(item["oracle_fallback_enabled"] is False for item in quality)
    assert all(item["maximum_oracle_fraction"] == 0.0 for item in quality)
    assert len({config["output_dir"] for config in configs.values()}) == 2
    assert "parent_retained_path" not in configs["normal"]["stage_c"]["semantic_coverage"]
    assert configs["hard"]["stage_c"]["semantic_coverage"]["parent_retained_path"].endswith(
        "/normal/stage_c_curated_chunks.jsonl"
    )


def test_confirmatory_training_uses_only_natural_token_arms() -> None:
    protocol = json.loads(
        (ROOT / "protocols/code_7m_normal_hard_confirmatory_v1.json").read_text(
            encoding="utf-8"
        )
    )
    materialization = json.loads(
        (
            ROOT
            / "protocols/code_7m_normal_hard_confirmatory_materialization_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert protocol["selector_boundary"]["equal_token_resampling"] is False
    assert set(materialization["arms"]) == {
        "raw_audited_natural",
        "normal_natural",
        "hard_natural",
    }
    assert "equal-token resampling" in materialization["packing"]["tail_rule"]


if __name__ == "__main__":
    test_luna_normal_and_hard_share_ranker_without_runtime_oracle()
    test_confirmatory_training_uses_only_natural_token_arms()
    print("[luna-final-curation-contract-v1] shared ranker and zero runtime oracle: pass")
