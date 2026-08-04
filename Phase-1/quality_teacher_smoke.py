from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, assert_never

from dotenv import load_dotenv

from quality_teacher_adapters import (
    NvidiaBuildBackend,
    TeacherModelAdapter,
)
from quality_teacher_local import LazyQwenLocalBackend
from quality_teacher_panel import (
    QualityPolicy,
    TeacherLocation,
    TeacherPanel,
    TeacherVote,
    load_teacher_panel,
)
from quality_teacher_runtime import (
    EvaluationUnit,
    PanelPolicyResult,
    TeacherAdapter,
    TeacherGenerationRequest,
    TeacherGenerationUnavailable,
    evaluate_panel_policy,
)


@dataclass(frozen=True, slots=True)
class SmokeContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return f"Quality teacher smoke contract failed: {self.detail}"


@dataclass(slots=True)
class AuditedAdapter:
    """Records non-sensitive hashes while delegating real model generation."""

    delegate: TeacherAdapter
    traces: list[dict[str, str | int | bool]] = field(default_factory=list)
    clock: Callable[[], float] = perf_counter

    def generate(self, request: TeacherGenerationRequest) -> str:
        started = self.clock()
        try:
            raw = self.delegate.generate(request)
        except TeacherGenerationUnavailable as error:
            self.traces.append(
                {
                    "teacher_id": request.teacher_id,
                    "policy_id": request.policy_id,
                    "unit_id": request.unit_id,
                    "pass_index": request.pass_index,
                    "schema_retry": request.schema_retry,
                    "elapsed_milliseconds": round((self.clock() - started) * 1000),
                    "status": "unavailable",
                    "error_reason": error.reason,
                }
            )
            raise
        self.traces.append(
            {
                "teacher_id": request.teacher_id,
                "policy_id": request.policy_id,
                "unit_id": request.unit_id,
                "pass_index": request.pass_index,
                "schema_retry": request.schema_retry,
                "elapsed_milliseconds": round((self.clock() - started) * 1000),
                "status": "success",
                "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            }
        )
        return raw


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy(panel: TeacherPanel, policy_id: str) -> QualityPolicy:
    matches = tuple(policy for policy in panel.policies if policy.policy_id == policy_id)
    if len(matches) != 1:
        raise SmokeContractError(detail=f"policy must exist exactly once: {policy_id}")
    return matches[0]


def _build_adapters(
    panel: TeacherPanel,
    local_model_path: Path,
) -> Mapping[str, AuditedAdapter]:
    local_backend: LazyQwenLocalBackend | None = None
    adapters: dict[str, AuditedAdapter] = {}
    for teacher in panel.teachers:
        match teacher.location:
            case TeacherLocation.NVIDIA_BUILD:
                environment_variable = teacher.api_key_environment_variable
                endpoint = teacher.endpoint_base_url
                if environment_variable is None or endpoint is None:
                    raise SmokeContractError(detail=f"hosted contract incomplete: {teacher.teacher_id}")
                api_key = os.environ.get(environment_variable, "")
                if (
                    teacher.request_timeout_seconds is None
                    or teacher.maximum_transport_retries is None
                ):
                    raise SmokeContractError(
                        detail=f"hosted transport contract incomplete: {teacher.teacher_id}"
                    )
                backend = NvidiaBuildBackend(
                    api_key=api_key,
                    base_url=endpoint,
                    timeout_seconds=teacher.request_timeout_seconds,
                    maximum_transport_retries=teacher.maximum_transport_retries,
                )
            case TeacherLocation.LOCAL:
                if local_backend is None:
                    local_backend = LazyQwenLocalBackend(local_model_path)
                backend = local_backend
            case unreachable:
                assert_never(unreachable)
        adapters[teacher.teacher_id] = AuditedAdapter(
            delegate=TeacherModelAdapter(
                teacher=teacher,
                backend=backend,
                maximum_new_tokens=teacher.maximum_new_tokens,
            )
        )
    return adapters


def _vote(vote: TeacherVote) -> dict[str, str | list[str]]:
    return {
        "teacher_id": vote.teacher_id,
        "policy_id": vote.policy_id,
        "decision": vote.decision.value,
        "reason_codes": list(vote.reason_codes),
    }


def _report(
    panel: TeacherPanel,
    policy: QualityPolicy,
    unit: EvaluationUnit,
    result: PanelPolicyResult,
    adapters: Mapping[str, AuditedAdapter],
    *,
    panel_path: Path,
    fixture_path: Path,
) -> dict[str, str | bool | list[dict[str, str | list[str]]] | list[dict[str, str | int | bool]]]:
    traces = [trace for adapter in adapters.values() for trace in adapter.traces]
    second_pass = [] if result.second_pass is None else [_vote(vote) for vote in result.second_pass]
    return {
        "schema_version": "quality-teacher-smoke-v1",
        "status": "smoke_only_not_promotion_evidence",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_panel_sha256": _sha256(panel_path),
        "fixture_sha256": _sha256(fixture_path),
        "unit_id": unit.unit_id,
        "policy_id": policy.policy_id,
        "panel_decision": result.decision.value,
        "runtime_activation": panel.runtime_activation,
        "teacher_output_alone_may_delete": panel.teacher_output_alone_may_delete,
        "first_pass": [_vote(vote) for vote in result.first_pass],
        "second_pass": second_pass,
        "generation_traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen three-teacher Quality smoke test.")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-id", default="q3_substantive_payload")
    args = parser.parse_args()

    if not load_dotenv(args.dotenv):
        raise SmokeContractError(detail=f"could not load dotenv file: {args.dotenv}")
    panel = load_teacher_panel(args.panel)
    unit = EvaluationUnit.model_validate_json(args.fixture.read_text(encoding="utf-8"))
    policy = _policy(panel, args.policy_id)
    adapters = _build_adapters(panel, args.local_model_path)
    result = evaluate_panel_policy(panel, adapters, policy, unit)
    report = _report(
        panel,
        policy,
        unit,
        result,
        adapters,
        panel_path=args.panel,
        fixture_path=args.fixture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "panel_decision": report["panel_decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
