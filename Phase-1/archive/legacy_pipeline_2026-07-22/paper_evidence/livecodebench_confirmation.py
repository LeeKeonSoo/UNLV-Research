from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

from data_eval_common import save_json, sha256_file


PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FREEZE: Final = PROJECT_DIR / "configs" / "code_livecodebench_pilot_v1.json"
DEFAULT_TRAINING_PLAN: Final = PROJECT_DIR / "configs" / "code_domain_natural_budget_protocol_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR: Final = PROJECT_DIR / "configs" / "code_livecodebench_confirmation_v1"
CURRENT_FRAMEWORK_TRAINING_OUTPUT: Final = (
    PROJECT_DIR / "outputs" / "code_domain_natural_budget_qwen3_4b" / "current_framework_rerun"
)
GPU_NAMES: Final = (
    "NVIDIA GeForce RTX 4060 Ti",
    "NVIDIA GeForce RTX 3070 Ti",
)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SeedGpuAssignment:
    seed: int
    required_gpu: str


@dataclass(frozen=True, slots=True)
class ConfirmationContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class ConfirmationReport(TypedDict):
    schema_version: str
    status: str
    source_freeze_path: str
    source_freeze_sha256: str
    task_bundle_path: str
    scheduled_seeds: list[int]
    assignments: list[JsonObject]
    selector_tuning_permission: bool
    utility_stage: str


def remaining_seed_schedule(
    *,
    training_seeds: tuple[int, ...],
    completed_seeds: tuple[int, ...],
) -> tuple[SeedGpuAssignment, ...]:
    completed = frozenset(completed_seeds)
    remaining = tuple(seed for seed in training_seeds if seed not in completed)
    if not remaining:
        raise ConfirmationContractError("No unobserved confirmation seeds remain")
    return tuple(
        SeedGpuAssignment(seed=seed, required_gpu=GPU_NAMES[index % len(GPU_NAMES)])
        for index, seed in enumerate(remaining)
    )


def _load_object(path: Path) -> JsonObject:
    payload = json.loads(path.read_text(encoding="utf-8"))
    match payload:
        case dict() as object_payload:
            return object_payload
        case _:
            raise ConfirmationContractError(f"Expected JSON object in {path}")


def _object_field(payload: JsonObject, key: str, context: str) -> JsonObject:
    value = payload.get(key)
    match value:
        case dict() as object_value:
            return object_value
        case _:
            raise ConfirmationContractError(f"Missing object field {context}.{key}")


def _seed_tuple(plan: JsonObject) -> tuple[int, ...]:
    training = _object_field(plan, "confirmatory_training_recipe", "training_plan")
    seeds = training.get("confirmatory_training_seeds")
    match seeds:
        case list() as seed_values if all(isinstance(value, int) for value in seed_values):
            return tuple(seed_values)
        case _:
            raise ConfirmationContractError("Training plan has no integer confirmatory seed schedule")


def _source_seed(source: JsonObject) -> int:
    execution = _object_field(source, "execution", "source_freeze")
    seed = execution.get("seed")
    match seed:
        case int() as integer_seed:
            return integer_seed
        case _:
            raise ConfirmationContractError("Source freeze has no integer execution seed")


def _task_bundle_path(source: JsonObject) -> str:
    dataset = _object_field(source, "dataset", "source_freeze")
    bundle = _object_field(dataset, "frozen_task_bundle", "source_freeze.dataset")
    path = bundle.get("path")
    match path:
        case str() as bundle_path:
            return bundle_path
        case _:
            raise ConfirmationContractError("Source freeze has no task-bundle path")


def _seed_freeze(source: JsonObject, assignment: SeedGpuAssignment, source_path: Path) -> JsonObject:
    copied = copy.deepcopy(source)
    execution = _object_field(copied, "execution", "confirmation_freeze")
    execution["seed"] = assignment.seed
    execution["required_gpu"] = assignment.required_gpu
    copied["schema_version"] = "code-livecodebench-confirmation-v1"
    copied["scope"] = "stage_c_independent_benchmark_confirmation"
    copied["confirmation"] = {
        "source_freeze_path": str(source_path),
        "source_freeze_sha256": sha256_file(source_path),
        "completed_seed": _source_seed(source),
        "selector_policy_changed": False,
        "selector_tuning_permission": False,
        "task_bundle_reused": True,
        "training_output_root": str(CURRENT_FRAMEWORK_TRAINING_OUTPUT),
    }
    return copied


def build_confirmation(
    *,
    source_freeze: Path,
    training_plan: Path,
    output_dir: Path,
) -> ConfirmationReport:
    source = _load_object(source_freeze)
    status = source.get("status")
    if status != "frozen_before_outcomes":
        raise ConfirmationContractError("Source freeze must be outcome-independent")
    schedule = remaining_seed_schedule(
        training_seeds=_seed_tuple(_load_object(training_plan)),
        completed_seeds=(_source_seed(source),),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for assignment in schedule:
        save_json(
            output_dir / f"seed{assignment.seed}.json",
            _seed_freeze(source, assignment, source_freeze),
        )
    return {
        "schema_version": "code-livecodebench-confirmation-manifest-v1",
        "status": "frozen_before_outcomes",
        "source_freeze_path": str(source_freeze),
        "source_freeze_sha256": sha256_file(source_freeze),
        "task_bundle_path": _task_bundle_path(source),
        "scheduled_seeds": [assignment.seed for assignment in schedule],
        "assignments": [
            {"seed": assignment.seed, "required_gpu": assignment.required_gpu}
            for assignment in schedule
        ],
        "selector_tuning_permission": False,
        "utility_stage": "Stage C only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-freeze", type=Path, default=DEFAULT_SOURCE_FREEZE)
    parser.add_argument("--training-plan", type=Path, default=DEFAULT_TRAINING_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report = build_confirmation(
        source_freeze=args.source_freeze,
        training_plan=args.training_plan,
        output_dir=args.output_dir,
    )
    save_json(args.output_dir / "confirmation_manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
