from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "207_build_code_natural_budget_stage_c_summary.py"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _load_module():
    spec = importlib.util.spec_from_file_location("code_natural_budget_stage_c_summary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_uses_explicit_current_run_inputs(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    data_dir = tmp_path / "frozen_data"
    nll_dir = tmp_path / "current_run" / "heldout_nll"
    evalplus_path = tmp_path / "current_evalplus.json"
    output_path = tmp_path / "current_summary.json"
    seeds = [101, 131]

    _write_json(
        plan_path,
        {
            "confirmatory_training_recipe": {"confirmatory_training_seeds": seeds},
            "utility_scope": "Stage C validation only; never selector objective",
        },
    )
    _write_json(
        data_dir / "natural_budget_arms_report.json",
        {
            "arms": {
                "raw_full_natural": {"records": 100, "token_proxy_count": 1000},
                "curated_v2_natural": {"records": 40, "token_proxy_count": 400},
            }
        },
    )
    _write_json(
        data_dir / "token_blocks" / "natural_budget_steps_report.json",
        {
            "packed_tokens_by_arm": {"raw_full_natural": 900, "curated_v2_natural": 360},
            "optimizer_steps_by_arm": {"raw_full_natural": 10, "curated_v2_natural": 4},
        },
    )
    _write_json(nll_dir / "base_no_update.json", {"mean_nll": 1.3, "tokens": 100})
    for seed, raw_nll, curated_nll in ((101, 1.2, 1.1), (131, 1.22, 1.12)):
        _write_json(nll_dir / f"raw_full_natural_seed{seed}.json", {"mean_nll": raw_nll})
        _write_json(nll_dir / f"curated_v2_natural_seed{seed}.json", {"mean_nll": curated_nll})
    _write_json(
        evalplus_path,
        {
            "arm_summaries": {
                "raw_full_natural": {
                    "macro_pass_rate": 0.5,
                    "datasets": {
                        "HumanEval+": {
                            "mean_pass_rate": 0.5,
                            "sample_std_pass_rate": 0.01,
                            "per_seed_pass_rate": {"101": 0.49, "131": 0.51},
                        }
                    },
                },
                "curated_v2_natural": {
                    "macro_pass_rate": 0.6,
                    "datasets": {
                        "HumanEval+": {
                            "mean_pass_rate": 0.6,
                            "sample_std_pass_rate": 0.01,
                            "per_seed_pass_rate": {"101": 0.59, "131": 0.61},
                        }
                    },
                },
            }
        },
    )

    report = _load_module().build(
        plan_path=plan_path,
        data_dir=data_dir,
        nll_dir=nll_dir,
        evalplus_report=evalplus_path,
        output_path=output_path,
    )

    assert report["seed_scope"] == seeds
    assert report["arms"]["curated_v2_natural"]["mean_nll"] == 1.11
    assert report["decision"] == "curated_better_than_raw_full_on_nll_and_evalplus"
    assert str(nll_dir / "curated_v2_natural_seed131.json") in report["source_sha256"]
    assert output_path.exists()


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_dir:
        test_build_uses_explicit_current_run_inputs(Path(temporary_dir))
    print("[code-natural-budget-stage-c-summary-inputs] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
