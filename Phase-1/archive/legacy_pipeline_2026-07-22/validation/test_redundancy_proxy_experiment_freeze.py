#!/usr/bin/env python3
"""Validate the frozen redundancy-saturation proxy experiment contract."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_eval_common import load_json, sha256_file


CONFIG = (
    ROOT / "configs" / "temporal_code_redundancy_proxy_experiment_qwen25_0p5b_v1.json"
)
REPORT = (
    ROOT
    / "validation"
    / "frozen_contracts"
    / "redundancy_proxy_experiment_freeze_report.json"
)


def main() -> int:
    plan = load_json(CONFIG)
    report = load_json(REPORT)
    assert plan["status"] == "frozen_before_proxy_training_outcomes"
    assert report["status"] == "redundancy_proxy_experiment_frozen"
    assert not plan["blockers"]
    assert not report["blockers"]
    assert report["config_sha256"] == sha256_file(CONFIG)

    model = plan["target_model"]
    assert model["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert model["revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert model["local_files_only"] is True
    assert set(model["snapshot_artifacts"]) == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }

    packing = plan["tokenization_and_packing"]
    assert packing["sequence_length"] == 1024
    assert packing["exact_train_tokens_per_arm"] == 327680
    assert packing["exact_train_blocks_per_arm"] == 320
    assert all(
        arm["raw_tokenizer_tokens_with_eos"] >= packing["exact_train_tokens_per_arm"]
        for arm in plan["arms"].values()
    )

    recipe = plan["training_recipe"]
    assert recipe["seeds"] == [11, 23, 37]
    assert recipe["optimizer_steps"] == 40
    assert (
        recipe["optimizer_steps"]
        * recipe["gradient_accumulation_steps"]
        * recipe["micro_batch_size"]
        * packing["sequence_length"]
        == packing["exact_train_tokens_per_arm"]
    )

    heldout = plan["heldout_nll"]
    assert heldout["exact_evaluation_tokens"] == 146432
    assert heldout["exact_evaluation_blocks"] == 143
    assert heldout["train_repository_overlap_count"] == 0
    assert set(heldout["allowed_content_types"]) == {"code", "test"}

    decision = plan["decision_contract"]
    assert decision["practical_absolute_nll_floor"] == 0.002
    assert "Stage C validation only" in plan["primary_comparison"]["utility_scope"]
    assert "abstain" in decision["mechanism_requirement"]
    assert "abstain" in decision["retention_requirement"]
    assert "Qwen3-4B" in decision["promotion_rule"]
    assert "outcomes" in " ".join(plan["forbidden_uses"])
    print("[redundancy-proxy-experiment] frozen model, tokens, seeds, heldout, and decision: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
