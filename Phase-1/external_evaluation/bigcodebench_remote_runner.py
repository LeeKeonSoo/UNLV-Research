"""Run BigCodeBench's official remote evaluator without local generation imports."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Final, Protocol

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError
import httpx


DEFAULT_ENDPOINT = "https://bigcode-bigcodebench-evaluator.hf.space/"
REMOTE_HTTP_TIMEOUT: Final = httpx.Timeout(
    connect=60.0,
    read=7_200.0,
    write=600.0,
    pool=60.0,
)
TRANSIENT_APP_ERROR: Final = re.compile(
    r"\b(?:500|502|503|504)\b|server error|bad gateway|service unavailable|gateway timeout",
    flags=re.IGNORECASE,
)


class RemoteClient(Protocol):
    def predict(self, **kwargs: object) -> tuple[object, object]: ...


def create_remote_client(endpoint: str) -> RemoteClient:
    """Create a Gradio client that permits the evaluator's long-running job."""
    return Client(endpoint, httpx_kwargs={"timeout": REMOTE_HTTP_TIMEOUT})


@dataclass(frozen=True, slots=True)
class RemoteEvaluationRequest:
    samples: Path
    result_path: Path | None = None
    pass_rate_path: Path | None = None
    split: str = "complete"
    subset: str = "full"
    pass_k: str = "1"
    parallel: int = -1
    calibrated: bool = True
    selective_evaluate: str = ""
    endpoint: str = DEFAULT_ENDPOINT
    max_attempts: int = 12
    retry_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.retry_seconds < 0:
            raise ValueError("retry_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class RemoteEvaluationArtifacts:
    result_path: Path
    pass_rate_path: Path


def _load_json_value(value: object) -> object:
    if isinstance(value, Path):
        return json.loads(value.read_text(encoding="utf-8"))
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    loaded = _load_json_value(value)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Official BigCodeBench {label} must be a JSON object")
    return loaded


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_remote_evaluation(
    request: RemoteEvaluationRequest,
    *,
    client_factory: Callable[[str], RemoteClient] = create_remote_client,
    file_handler: Callable[[Path], object] = lambda path: handle_file(str(path)),
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RemoteEvaluationArtifacts:
    if not request.samples.is_file():
        raise FileNotFoundError(request.samples)

    result_path = request.result_path or Path(
        str(request.samples).replace(".jsonl", "_eval_results.json")
    )
    pass_rate_path = request.pass_rate_path or Path(
        str(request.samples).replace(".jsonl", "_pass_at_k.json")
    )
    if result_path.is_file() and pass_rate_path.is_file():
        return RemoteEvaluationArtifacts(result_path, pass_rate_path)

    retryable_errors = (
        CancelledError,
        OSError,
        TimeoutError,
        ValueError,
        httpx.HTTPError,
    )
    for attempt in range(1, request.max_attempts + 1):
        try:
            client = client_factory(request.endpoint)
            raw_results, raw_pass_rate = client.predict(
                split=request.split,
                subset=request.subset,
                samples=file_handler(request.samples),
                pass_k=request.pass_k,
                parallel=request.parallel,
                min_time_limit=1,
                max_as_limit=30 * 1024,
                max_data_limit=30 * 1024,
                max_stack_limit=10,
                calibrated=request.calibrated,
                check_gt_only=False,
                no_gt=False,
                selective_evaluate=request.selective_evaluate,
                api_name="/predict",
            )
            break
        except AppError as error:
            if TRANSIENT_APP_ERROR.search(str(error)) is None:
                raise
            if attempt == request.max_attempts:
                raise
            print(
                f"BigCodeBench endpoint unavailable; retrying in "
                f"{request.retry_seconds:.0f}s "
                f"({attempt}/{request.max_attempts})",
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            sleep_fn(request.retry_seconds)
        except retryable_errors as error:
            if attempt == request.max_attempts:
                raise
            print(
                f"BigCodeBench endpoint unavailable; retrying in "
                f"{request.retry_seconds:.0f}s "
                f"({attempt}/{request.max_attempts})",
                f"error={type(error).__name__}: {error}",
                flush=True,
            )
            sleep_fn(request.retry_seconds)
    results = _require_mapping(raw_results, label="results")
    pass_rate = _require_mapping(raw_pass_rate, label="pass rate")
    if "pass@1" not in pass_rate:
        raise ValueError("Official BigCodeBench response omitted pass@1")

    _write_json(result_path, results)
    _write_json(pass_rate_path, pass_rate)
    return RemoteEvaluationArtifacts(result_path, pass_rate_path)


def _parse_args() -> RemoteEvaluationRequest:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--split", default="complete")
    parser.add_argument("--subset", default="full")
    parser.add_argument("--pass-k", default="1")
    parser.add_argument("--parallel", type=int, default=-1)
    parser.add_argument("--calibrated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--selective-evaluate", default="")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    args = parser.parse_args()
    return RemoteEvaluationRequest(
        samples=args.samples,
        split=args.split,
        subset=args.subset,
        pass_k=args.pass_k,
        parallel=args.parallel,
        calibrated=args.calibrated,
        selective_evaluate=args.selective_evaluate,
        endpoint=args.endpoint,
        max_attempts=args.max_attempts,
        retry_seconds=args.retry_seconds,
    )


def main() -> int:
    artifacts = run_remote_evaluation(_parse_args())
    print(
        json.dumps(
            {
                "result_path": str(artifacts.result_path),
                "pass_rate_path": str(artifacts.pass_rate_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
