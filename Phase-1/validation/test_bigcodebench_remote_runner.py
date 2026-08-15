import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from gradio_client.exceptions import AppError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_evaluation.bigcodebench_remote_runner import (
    REMOTE_HTTP_TIMEOUT,
    RemoteEvaluationRequest,
    run_remote_evaluation,
)


class FakeClient:
    def predict(self, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        assert kwargs["api_name"] == "/predict"
        return (
            {"eval": {"BigCodeBench/0": [{"status": "pass"}]}},
            {
                "pass@1": 0.25,
                "gt_pass_rate": 1.0,
                "failed_tasks": [],
            },
        )


def test_remote_runner_persists_official_results_without_bigcodebench_cli() -> None:
    assert REMOTE_HTTP_TIMEOUT.read == 7_200.0
    with TemporaryDirectory() as directory:
        samples = Path(directory) / "samples.jsonl"
        samples.write_text('{"task_id":"BigCodeBench/0","solution":"pass"}\n')
        request = RemoteEvaluationRequest(samples=samples)

        artifacts = run_remote_evaluation(
            request,
            client_factory=lambda _endpoint: FakeClient(),
            file_handler=lambda path: str(path),
        )

        result = json.loads(artifacts.result_path.read_text(encoding="utf-8"))
        pass_rate = json.loads(artifacts.pass_rate_path.read_text(encoding="utf-8"))
        assert result["eval"]["BigCodeBench/0"][0]["status"] == "pass"
        assert pass_rate["pass@1"] == 0.25


def test_remote_runner_retries_temporary_endpoint_failure() -> None:
    attempts = 0
    sleeps: list[float] = []

    def client_factory(_endpoint: str) -> FakeClient:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("temporary 503")
        return FakeClient()

    with TemporaryDirectory() as directory:
        samples = Path(directory) / "samples.jsonl"
        samples.write_text('{"task_id":"BigCodeBench/0","solution":"pass"}\n')
        request = RemoteEvaluationRequest(
            samples=samples,
            max_attempts=2,
            retry_seconds=3.0,
        )

        run_remote_evaluation(
            request,
            client_factory=client_factory,
            file_handler=lambda path: str(path),
            sleep_fn=sleeps.append,
        )

    assert attempts == 2
    assert sleeps == [3.0]


def test_remote_runner_writes_to_explicit_chunk_artifact_paths() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        samples = root / "full_samples.jsonl"
        samples.write_text('{"task_id":"BigCodeBench/0","solution":"pass"}\n')
        result_path = root / "chunks" / "chunk_000_eval_results.json"
        pass_rate_path = root / "chunks" / "chunk_000_pass_at_k.json"
        request = RemoteEvaluationRequest(
            samples=samples,
            result_path=result_path,
            pass_rate_path=pass_rate_path,
            selective_evaluate="0",
        )

        artifacts = run_remote_evaluation(
            request,
            client_factory=lambda _endpoint: FakeClient(),
            file_handler=lambda path: str(path),
        )

        assert artifacts.result_path == result_path
        assert artifacts.pass_rate_path == pass_rate_path
        assert result_path.is_file()
        assert pass_rate_path.is_file()


def test_remote_runner_does_not_retry_deterministic_evaluator_rejection() -> None:
    attempts = 0

    class RejectingClient:
        def predict(self, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            nonlocal attempts
            attempts += 1
            raise AppError("unknown task ID")

    with TemporaryDirectory() as directory:
        samples = Path(directory) / "samples.jsonl"
        samples.write_text('{"task_id":"BigCodeBench/0","solution":"pass"}\n')
        request = RemoteEvaluationRequest(samples=samples, max_attempts=5)

        try:
            run_remote_evaluation(
                request,
                client_factory=lambda _endpoint: RejectingClient(),
                file_handler=lambda path: str(path),
                sleep_fn=lambda _seconds: None,
            )
        except AppError:
            pass
        else:
            raise AssertionError("deterministic evaluator rejection must propagate")

    assert attempts == 1


def test_remote_runner_retries_transient_evaluator_server_error() -> None:
    attempts = 0

    class TransientClient:
        def predict(self, **_kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise AppError("502 Server Error: Bad Gateway")
            return FakeClient().predict(**_kwargs)

    with TemporaryDirectory() as directory:
        samples = Path(directory) / "samples.jsonl"
        samples.write_text('{"task_id":"BigCodeBench/0","solution":"pass"}\n')
        request = RemoteEvaluationRequest(samples=samples, max_attempts=2)

        run_remote_evaluation(
            request,
            client_factory=lambda _endpoint: TransientClient(),
            file_handler=lambda path: str(path),
            sleep_fn=lambda _seconds: None,
        )

    assert attempts == 2


if __name__ == "__main__":
    test_remote_runner_persists_official_results_without_bigcodebench_cli()
    test_remote_runner_retries_temporary_endpoint_failure()
    test_remote_runner_writes_to_explicit_chunk_artifact_paths()
    test_remote_runner_does_not_retry_deterministic_evaluator_rejection()
    test_remote_runner_retries_transient_evaluator_server_error()
    print("BigCodeBench remote runner contract passed")
