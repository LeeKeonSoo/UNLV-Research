from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from archive.temporal_code.code_evalplus_samples import (
    _device_summary,
    _load_model,
    _load_tokenizer,
    _optimizer_steps,
    _run_dir,
    _set_seed,
)
from data_eval_common import load_json, save_json, sha256_file
from paper_evidence.livecodebench_freeze import DEFAULT_LCB_REPO

PROJECT_DIR: Final = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE: Final = PROJECT_DIR / "configs" / "code_livecodebench_pilot_v1.json"
DEFAULT_PLAN: Final = PROJECT_DIR / "configs" / "code_domain_natural_budget_protocol_qwen3_4b_v1.json"
DEFAULT_TRAINING_OUTPUT: Final = PROJECT_DIR / "outputs" / "code_domain_natural_budget_qwen3_4b" / "current_framework_rerun"
DEFAULT_OUTPUT: Final = PROJECT_DIR / "outputs" / "code_livecodebench_pilot_qwen3_4b"
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class GenerationReport(TypedDict):
    schema_version: str
    status: str
    arm: str
    seed: int
    task_count: int
    freeze_sha256: str
    plan_sha256: str
    generation_path: str
    generation_sha256: str
    adapter_path: str | None
    adapter_sha256: str | None
    device_summary: JsonValue
    elapsed_seconds: float
    utility_stage: str
    selector_tuning_permission: bool


@dataclass(frozen=True, slots=True)
class PromptRecord:
    question_id: str
    question_content: str
    starter_code: str


@dataclass(frozen=True, slots=True)
class RunInputs:
    arm: str
    freeze_path: Path
    plan_path: Path
    lcb_repo: Path
    training_output: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class RunContractError(RuntimeError):
    detail: str

    def __str__(self) -> str:
        return self.detail


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, token_sequences: tuple[tuple[int, ...], ...]) -> None:
        self._token_sequences = token_sequences

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
        **_: str | int | float | bool | torch.Tensor | None,
    ) -> bool:
        del scores
        return all(
            any(
                len(sequence) <= row.shape[0]
                and tuple(row[-len(sequence) :].tolist()) == sequence
                for sequence in self._token_sequences
            )
            for row in input_ids
        )


def trim_generic_base_completion(text: str) -> str:
    cut = text.find("###")
    completion = text if cut < 0 else text[:cut]
    return completion.rstrip() + "\n"


def _load_prompts(
    freeze: dict,
    lcb_repo: Path,
) -> tuple[tuple[str, str], ...]:
    task_bundle = Path(str(freeze["dataset"]["frozen_task_bundle"]["path"]))
    dataset = load_json(task_bundle)
    records = tuple(
        PromptRecord(
            question_id=str(row["question_id"]),
            question_content=str(row["question_content"]),
            starter_code=str(row["starter_code"]),
        )
        for row in dataset
    )
    sys.path.insert(0, str(lcb_repo))
    with chdir(lcb_repo):
        from lcb_runner.prompts.code_generation import (
            get_base_model_question_template_answer,
        )

        prompts = tuple(
            (
                record.question_id,
                get_base_model_question_template_answer(record),
            )
            for record in records
        )
    return tuple(sorted(prompts))


def _partial_records(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        rows = (json.loads(line) for line in handle if line.strip())
        return {str(row["question_id"]): row for row in rows}


def _arm_suffix(arm: str, seed: int) -> str:
    return "base" if arm == "base_no_update" else f"seed{seed}"


@torch.no_grad()
def generate_arm(
    inputs: RunInputs,
) -> GenerationReport:
    freeze = load_json(inputs.freeze_path)
    if freeze["status"] != "frozen_before_outcomes" or not freeze["execution"]["allowed"]:
        raise RunContractError("LiveCodeBench pilot execution is not allowed by the freeze")
    allowed_arms = tuple(str(value) for value in freeze["execution"]["arms"])
    if inputs.arm not in allowed_arms:
        raise RunContractError(f"Arm is outside the frozen contract: {inputs.arm}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RunContractError("Exactly one visible CUDA device is required")
    required_gpu = str(freeze["execution"]["required_gpu"])
    if torch.cuda.get_device_name(0) != required_gpu:
        raise RunContractError(
            f"Required GPU is {required_gpu}; got {torch.cuda.get_device_name(0)}"
        )

    seed = int(freeze["execution"]["seed"])
    _set_seed(seed)
    prompts = _load_prompts(freeze, inputs.lcb_repo)
    inputs.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _arm_suffix(inputs.arm, seed)
    partial_path = inputs.output_dir / f"{inputs.arm}_{suffix}_partial.jsonl"
    final_path = inputs.output_dir / f"{inputs.arm}_{suffix}_generations.json"
    manifest_path = inputs.output_dir / f"{inputs.arm}_{suffix}_manifest.json"
    completed = _partial_records(partial_path)
    plan = load_json(inputs.plan_path)
    tokenizer = _load_tokenizer(plan, allow_download=False)
    model = _load_model(
        plan,
        inputs.training_output,
        inputs.arm,
        seed,
        allow_download=False,
    )
    stop_sequences = tuple(
        tuple(tokenizer.encode(value, add_special_tokens=False))
        for value in freeze["execution"]["stop_strings"]
    )
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])
    started = time.time()

    with partial_path.open("a", encoding="utf-8") as handle:
        for index, (question_id, prompt) in enumerate(prompts, start=1):
            if question_id in completed:
                continue
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(0)
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(freeze["execution"]["max_new_tokens"]),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            row = {
                "question_id": question_id,
                "code": trim_generic_base_completion(raw_output),
                "raw_output": raw_output,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed[question_id] = row
            print(
                json.dumps(
                    {"arm": inputs.arm, "completed": index, "total": len(prompts)}
                ),
                flush=True,
            )

    ordered = [completed[question_id] for question_id, _ in prompts]
    save_json(
        final_path,
        [
            {"question_id": row["question_id"], "code_list": [row["code"]]}
            for row in ordered
        ],
    )
    adapter = None
    if inputs.arm != "base_no_update":
        adapter = _run_dir(
            inputs.training_output,
            inputs.arm,
            seed,
            _optimizer_steps(plan, inputs.arm),
        )
    manifest: GenerationReport = {
        "schema_version": "code-livecodebench-pilot-generation-v1",
        "status": "generation_completed",
        "arm": inputs.arm,
        "seed": seed,
        "task_count": len(ordered),
        "freeze_sha256": sha256_file(inputs.freeze_path),
        "plan_sha256": sha256_file(inputs.plan_path),
        "generation_path": str(final_path),
        "generation_sha256": sha256_file(final_path),
        "adapter_path": str(adapter) if adapter else None,
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors") if adapter else None,
        "device_summary": _device_summary(),
        "elapsed_seconds": round(time.time() - started, 3),
        "utility_stage": "Stage C only",
        "selector_tuning_permission": False,
    }
    save_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--lcb-repo", type=Path, default=DEFAULT_LCB_REPO)
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_arm(RunInputs(
        arm=args.arm,
        freeze_path=args.freeze,
        plan_path=args.plan,
        lcb_repo=args.lcb_repo,
        training_output=args.training_output,
        output_dir=args.output_dir,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
