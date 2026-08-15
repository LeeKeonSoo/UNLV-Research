from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "tmp" / "run_two_seed_evalplus_benchmark_major.ps1"


def test_evalplus_queue_completes_one_benchmark_across_all_arms() -> None:
    # Given: the two-seed EvalPlus execution queue.
    source = QUEUE.read_text(encoding="utf-8")

    # When: its loop nesting and commands are inspected.
    dataset_loop = source.index('foreach ($Dataset in @("HumanEval+", "MBPP+"))')
    job_loop = source.index("foreach ($Job in $Jobs)")

    # Then: benchmark is the outer unit, with safe generation and official scoring.
    assert dataset_loop < job_loop
    assert '"--dataset", $Dataset' in source
    assert '"--batch-size", "1"' in source
    assert 'external_evaluation.evalplus_windows_runner' in source
    assert 'ValidateSet("first", "second")' in source
    assert "$Jobs = $JobGroups[$JobGroup]" in source
    assert "function Wait-ForBenchmarkResults" in source
    assert "Wait-ForBenchmarkResults $DatasetSlug" in source


if __name__ == "__main__":
    test_evalplus_queue_completes_one_benchmark_across_all_arms()
    print("EvalPlus benchmark-major queue contract passed")
