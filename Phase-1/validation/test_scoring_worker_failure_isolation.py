from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tmp" / "run_two_seed_scoring_worker.ps1"


def test_scoring_worker_isolates_each_official_benchmark_failure() -> None:
    source = WORKER.read_text(encoding="utf-8")

    assert "function Invoke-IsolatedNative" in source
    assert '"evalplus-${Dataset}"' in source
    assert '"bigcodebench-sanitize"' in source
    assert '"bigcodebench-evaluate"' in source
    assert '"external_evaluation.bigcodebench_remote_runner"' in source
    assert '"cruxeval-${Mode}"' in source
    assert '"external_evaluation.cruxeval_windows_runner"' in source
    assert '"ds1000"' in source
    assert 'Join-Path $Results "status"' in source
    assert source.rfind('"bigcodebench-evaluate"') > source.rfind('"ds1000"')
    assert "$PhaseRoot = Split-Path -Parent $PSScriptRoot" in source
    assert '$env:PYTHONPATH = "$PhaseRoot;$BigCodeRoot"' in source


if __name__ == "__main__":
    test_scoring_worker_isolates_each_official_benchmark_failure()
    print("scoring worker failure-isolation contract passed")
