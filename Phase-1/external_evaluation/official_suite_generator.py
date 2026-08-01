#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Final, Iterable

import torch

from external_evaluation.evalplus_generator import RUN_ROOT, load_model, trim_completion


THIRD_PARTY_ROOT: Final = Path("D:/UNLV-Research/third_party")
HF_HUB_ROOT: Final = Path("D:/UNLV-Research/hf_cache/hub")
HF_DATASETS_ROOT: Final = Path("D:/UNLV-Research/hf_datasets_cache")
SAMPLE_DIRECTORY: Final = "official_suite_samples"


def livecodebench_release_files(data_root: Path, release: str) -> list[Path]:
    release_file_count = {
        "release_v1": 1,
        "release_v2": 2,
        "release_v3": 3,
        "release_v4": 4,
        "release_v5": 5,
        "release_v6": 6,
    }
    try:
        file_count = release_file_count[release]
    except KeyError as exc:
        raise ValueError(f"Unsupported LiveCodeBench release: {release}") from exc
    return [data_root / ("test.jsonl" if index == 1 else f"test{index}.jsonl") for index in range(1, file_count + 1)]


def bigcodebench_parquet_path(data_root: Path) -> Path:
    return data_root / "data" / "v0.1.4-00000-of-00001.parquet"


def resolve_livecodebench_data_root() -> Path:
    override = os.environ.get("LIVECODEBENCH_DATA_ROOT")
    if override:
        return Path(override)
    snapshots = HF_HUB_ROOT / "datasets--livecodebench--code_generation_lite" / "snapshots"
    candidates = sorted((path for path in snapshots.glob("*") if path.is_dir()), reverse=True)
    for candidate in candidates:
        if all(path.is_file() for path in livecodebench_release_files(candidate, "release_v6")):
            return candidate
    raise FileNotFoundError(f"LiveCodeBench release_v6 files are unavailable under {snapshots}")


def resolve_bigcodebench_data_root() -> Path:
    override = os.environ.get("BIGCODEBENCH_DATA_ROOT")
    if override:
        return Path(override)
    snapshots = HF_HUB_ROOT / "datasets--bigcode--bigcodebench" / "snapshots"
    candidates = sorted((path for path in snapshots.glob("*") if path.is_dir()), reverse=True)
    for candidate in candidates:
        if bigcodebench_parquet_path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"BigCodeBench v0.1.4 parquet is unavailable under {snapshots}")


def output_path(run_root: Path, suite: str, arm: str, seed: int | None) -> Path:
    suffix = "base" if seed is None else f"seed{seed}"
    extension = ".json" if suite == "livecodebench" else ".jsonl"
    return run_root / SAMPLE_DIRECTORY / f"{suite}_{arm}_{suffix}{extension}"


def add_source_path(name: str) -> None:
    path = THIRD_PARTY_ROOT / name
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)


def generate_text(model: object, tokenizer: object, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(0)
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    width = encoded["input_ids"].shape[1]
    return trim_completion(tokenizer.decode(generated[0][width:], skip_special_tokens=True))


def livecodebench_records(model: object, tokenizer: object, max_new_tokens: int) -> list[dict[str, object]]:
    add_source_path("LiveCodeBench")
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.prompts.code_generation import get_base_model_question_template_answer

    problems = []
    for path in livecodebench_release_files(resolve_livecodebench_data_root(), "release_v6"):
        with path.open(encoding="utf-8") as handle:
            problems.extend(CodeGenerationProblem(**json.loads(line)) for line in handle)
    return [
        {
            "question_id": problem.question_id,
            "code_list": [generate_text(model, tokenizer, get_base_model_question_template_answer(problem), max_new_tokens)],
        }
        for problem in problems
    ]


def bigcodebench_records(model: object, tokenizer: object, max_new_tokens: int) -> Iterable[dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "parquet",
        data_files=str(bigcodebench_parquet_path(resolve_bigcodebench_data_root())),
        split="train",
        cache_dir=str(HF_DATASETS_ROOT),
    )
    for problem in dataset:
        task_id = str(problem["task_id"])
        yield {
            "task_id": task_id,
            "completion": generate_text(model, tokenizer, problem["complete_prompt"], max_new_tokens),
            "_identifier": task_id,
        }


def ds1000_records(model: object, tokenizer: object, max_new_tokens: int) -> Iterable[dict[str, str]]:
    path = THIRD_PARTY_ROOT / "DS-1000" / "data" / "ds1000.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            yield {"code": generate_text(model, tokenizer, item["prompt"], max_new_tokens)}


def ojbench_records(model: object, tokenizer: object, max_new_tokens: int) -> Iterable[dict[str, object]]:
    path = THIRD_PARTY_ROOT / "OJBench_testdata" / "prompts" / "full.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            item["content"] = generate_text(model, tokenizer, item["prompt"], max_new_tokens)
            yield item


def preflight_suite(suite: str) -> None:
    required_paths = {
        "livecodebench": livecodebench_release_files(resolve_livecodebench_data_root(), "release_v6"),
        "bigcodebench": [bigcodebench_parquet_path(resolve_bigcodebench_data_root())],
        "ds1000": [THIRD_PARTY_ROOT / "DS-1000" / "data" / "ds1000.jsonl.gz"],
        "ojbench": [THIRD_PARTY_ROOT / "OJBench_testdata" / "prompts" / "full.jsonl"],
    }[suite]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{suite} preflight failed; missing required files: {missing}")


def write_records(suite: str, output: Path, model: object, tokenizer: object, max_new_tokens: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    match suite:
        case "livecodebench":
            output.write_text(json.dumps(livecodebench_records(model, tokenizer, max_new_tokens)), encoding="utf-8")
        case "bigcodebench":
            records = bigcodebench_records(model, tokenizer, max_new_tokens)
            with output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        case "ds1000":
            records = ds1000_records(model, tokenizer, max_new_tokens)
            with output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        case "ojbench":
            records = ojbench_records(model, tokenizer, max_new_tokens)
            with output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
        case unsupported:
            raise ValueError(f"Unsupported suite: {unsupported}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official-evaluator inputs from frozen QLoRA checkpoints.")
    parser.add_argument("--suite", required=True, choices=("livecodebench", "bigcodebench", "ds1000", "ojbench"))
    parser.add_argument("--arm", required=True, choices=("base_no_update", "raw_safe_natural", "curated_natural"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.arm == "base_no_update" and args.seed is not None:
        parser.error("base_no_update does not accept --seed")
    if args.arm != "base_no_update" and args.seed is None:
        parser.error("adapter arms require --seed")
    if not torch.cuda.is_available():
        raise RuntimeError("Official-suite generation requires CUDA")
    preflight_suite(args.suite)
    protocol = json.loads((Path(__file__).resolve().parents[1] / "protocols" / "code_evalplus_natural_3arm_qwen3_4b_v1.json").read_text(encoding="utf-8"))
    model, tokenizer = load_model(protocol, args.arm, args.seed)
    output = output_path(RUN_ROOT, args.suite, args.arm, args.seed)
    write_records(args.suite, output, model, tokenizer, args.max_new_tokens)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
