from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "145_build_code_domain_evalplus_guardrail_report.py"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _load_module():
    spec = importlib.util.spec_from_file_location("code_5m_external_evidence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_describes_complete_natural_budget_evidence_without_stage_c_guardrails(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    split_path = tmp_path / "split.json"
    input_report_path = tmp_path / "inputs.json"
    run_dir = tmp_path / "runs"
    output_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    seeds = [11, 23, 37]
    _write_json(
        plan_path,
        {
            "training_arms": ["raw_safe_natural", "curated_natural"],
            "training_recipe": {
                "development_training_seeds": seeds,
                "optimizer_steps_by_arm": {"raw_safe_natural": 429, "curated_natural": 156},
            },
            "frozen_inputs": {"materialization_report": str(input_report_path)},
            "utility_scope": "External validation only; never selector objective",
        },
    )
    _write_json(split_path, {"summary": {"suite_split_counts": {"HumanEval+/development": 2, "MBPP+/development": 2}}})
    _write_json(
        input_report_path,
        {"arms": {"raw_safe_natural": {"packed_tokens": 1000}, "curated_natural": {"packed_tokens": 400}}},
    )
    _write_json(run_dir / "heldout_nll" / "base_no_update.json", {"mean_nll": 1.1, "tokens": 200})
    for seed, raw_nll, curated_nll in ((11, 1.09, 1.05), (23, 1.10, 1.06), (37, 1.08, 1.04)):
        _write_json(run_dir / "heldout_nll" / f"raw_safe_natural_seed{seed}.json", {"mean_nll": raw_nll, "tokens": 200})
        _write_json(run_dir / "heldout_nll" / f"curated_natural_seed{seed}.json", {"mean_nll": curated_nll, "tokens": 200})
        for dataset, raw_rate, curated_rate in (("humaneval", 0.4, 0.5), ("mbpp", 0.5, 0.6)):
            _write_json(
                run_dir / "evalplus_guardrail" / "results" / f"{dataset}_raw_safe_natural_seed{seed}_eval.json",
                {"status": "evalplus_samples_evaluated", "task_count": 2, "pass_count": 1, "pass_rate": raw_rate},
            )
            _write_json(
                run_dir / "evalplus_guardrail" / "results" / f"{dataset}_curated_natural_seed{seed}_eval.json",
                {"status": "evalplus_samples_evaluated", "task_count": 2, "pass_count": 1, "pass_rate": curated_rate},
            )
    for dataset, rate in (("humaneval", 0.45), ("mbpp", 0.42)):
        _write_json(
            run_dir / "evalplus_guardrail" / "results" / f"{dataset}_base_no_update_base_eval.json",
            {"status": "evalplus_samples_evaluated", "task_count": 2, "pass_count": 1, "pass_rate": rate},
        )
    report = _load_module().build(
        plan_path=plan_path,
        split_path=split_path,
        run_dir=run_dir,
        output_path=output_path,
        markdown_path=markdown_path,
    )

    assert report["status"] == "external_evidence_complete"
    assert report["expected_suite_task_counts"] == {"humaneval": 2, "mbpp": 2}
    assert set(report["arms"]) == {"base_no_update", "raw_safe_natural", "curated_natural"}
    assert abs(report["raw_vs_curated"]["macro_delta_curated_minus_raw"] - 0.1) < 1e-12
    for label, expected in {"HumanEval+": -0.05, "MBPP+": 0.08, "macro": 0.015}.items():
        assert abs(report["base_retention"]["raw_minus_base"][label] - expected) < 1e-12
    for label, expected in {"HumanEval+": 0.05, "MBPP+": 0.18, "macro": 0.115}.items():
        assert abs(report["base_retention"]["curated_minus_base"][label] - expected) < 1e-12
    assert report["base_retention"]["uncertainty_note"] == "Base has one deterministic result; no Base uncertainty or significance claim is made."
    assert "universal" in report["claim_boundary"].lower()
    assert str(plan_path) in report["source_sha256"]
    assert str(split_path) in report["source_sha256"]
    assert str(input_report_path) in report["source_sha256"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Base-retention interpretation" in markdown
    assert "Raw-minus-Base: HumanEval+ `-0.050000`; MBPP+ `0.080000`; macro `0.015000`." in markdown
    assert "Curated-minus-Base: HumanEval+ `0.050000`; MBPP+ `0.180000`; macro `0.115000`." in markdown
    assert "MBPP+ Base-to-Curated change: `0.180000`." in markdown
    assert output_path.exists()


def test_build_marks_mismatched_evalplus_task_count_incomplete(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    split_path = tmp_path / "split.json"
    input_report_path = tmp_path / "inputs.json"
    run_dir = tmp_path / "runs"
    output_path = tmp_path / "report.json"
    seeds = [11, 23, 37]
    _write_json(plan_path, {"training_recipe": {"development_training_seeds": seeds, "optimizer_steps_by_arm": {"raw_safe_natural": 1, "curated_natural": 1}}, "frozen_inputs": {"materialization_report": str(input_report_path)}, "utility_scope": "External validation only"})
    _write_json(split_path, {"summary": {"suite_split_counts": {"HumanEval+/development": 2, "MBPP+/development": 2}}})
    _write_json(input_report_path, {"arms": {"raw_safe_natural": {"packed_tokens": 1}, "curated_natural": {"packed_tokens": 1}}})
    _write_json(run_dir / "heldout_nll" / "base_no_update.json", {"mean_nll": 1.0, "tokens": 1})
    for arm, arm_seeds in (("base_no_update", [None]), ("raw_safe_natural", seeds), ("curated_natural", seeds)):
        for seed in arm_seeds:
            suffix = "base" if seed is None else f"seed{seed}"
            _write_json(run_dir / "heldout_nll" / ("base_no_update.json" if seed is None else f"{arm}_seed{seed}.json"), {"mean_nll": 1.0, "tokens": 1})
            for dataset in ("humaneval", "mbpp"):
                task_count = 1 if (dataset, arm, seed) == ("mbpp", "raw_safe_natural", 11) else 2
                _write_json(run_dir / "evalplus_guardrail" / "results" / f"{dataset}_{arm}_{suffix}_eval.json", {"status": "evalplus_samples_evaluated", "task_count": task_count, "pass_count": 1, "pass_rate": 0.5})

    report = _load_module().build(plan_path=plan_path, split_path=split_path, run_dir=run_dir, output_path=output_path)

    assert report["status"] == "external_evidence_incomplete"
    assert any("task_count_mismatch" in blocker and "mbpp_raw_safe_natural_seed11_eval.json" in blocker for blocker in report["blockers"])


def main() -> int:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temporary_dir:
        test_build_describes_complete_natural_budget_evidence_without_stage_c_guardrails(Path(temporary_dir))
        test_build_marks_mismatched_evalplus_task_count_incomplete(Path(temporary_dir) / "mismatched_task_count")
    print("[code-5m-natural-budget-external-evidence-report] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
