#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.math_lm_eval_runner import (
    ARMS,
    EvaluationIdentity,
    clear_pretrained_generation_limit,
    output_path,
    resolve_tasks,
    validate_batch_size,
)


class FakeGenerationConfig:
    max_new_tokens: int | None = 2048


class FakeModel:
    generation_config = FakeGenerationConfig()


def main() -> int:
    assert "framework_math_natural" in ARMS
    assert "random_math_hard_matched" in ARMS
    assert "data_juicer_math_natural" in ARMS
    root = Path("D:/math-runs")
    assert output_path(root, EvaluationIdentity("base_no_update", None, None, None)) == (
        root / "math_lm_eval" / "base_no_update_base.json"
    )
    assert output_path(
        root,
        EvaluationIdentity("hard_math_natural", 101, 2, "hendrycks_math500"),
    ) == (
        root
        / "math_lm_eval"
        / "hard_math_natural_seed101_hendrycks_math500_limit2.json"
    )
    assert resolve_tasks(None) == ("gsm8k_cot_zeroshot", "hendrycks_math500")
    assert resolve_tasks("hendrycks_math500") == ("hendrycks_math500",)
    model = FakeModel()
    clear_pretrained_generation_limit(model)
    assert model.generation_config.max_new_tokens is None
    assert validate_batch_size(4) == 4
    assert validate_batch_size("4") == 4
    try:
        validate_batch_size(0)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-positive evaluation batch size must fail")
    print("[math-lm-eval-runner] output and task contracts: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
