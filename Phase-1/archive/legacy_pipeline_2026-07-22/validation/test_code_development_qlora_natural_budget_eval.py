#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.temporal_code.code_development_qlora import _evaluation_optimizer_steps
from data_eval_common import load_json


def main() -> int:
    plan = load_json(ROOT / "configs" / "raw_corpus_matrix_natural_budget_execution_qwen3_4b_v2.json")

    assert _evaluation_optimizer_steps(plan, "base_no_update") == 0
    assert _evaluation_optimizer_steps(plan, "curated_natural") == 27
    assert _evaluation_optimizer_steps(plan, "raw_mixed_all_natural") == 42

    print("[code-development-qlora-natural-budget-eval] base and arm steps: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
