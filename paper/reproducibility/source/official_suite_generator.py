#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Any, Final, Iterable, Literal, Sequence, TypeVar

# On Windows, Arrow DLLs must be initialized before Torch's CUDA DLLs.
import pyarrow.parquet as pq
import torch

from external_evaluation.evalplus_generator import (
    ARMS,
    INPUT_REPORT_PATH,
    PROTOCOL_PATH,
    benchmark_root,
    load_json,
    load_model,
    resolve_model_run,
    trim_completion,
)
from external_evaluation.runtime_paths import BenchmarkWorkerPaths


SAMPLE_DIRECTORY: Final = "official_suite_samples"
SUITES: Final = ("bigcodebench", "cruxeval_input", "cruxeval_output", "ds1000")
CruxMode = Literal["input", "output"]
CruxPromptStyle = Literal["canonical_direct", "answer_prefix_v1"]
RecordT = TypeVar("RecordT")


def bigcodebench_parquet_path(data_root: Path) -> Path:
    return data_root / "data" / "v0.1.4-00000-of-00001.parquet"


def load_bigcodebench_problems(parquet_path: Path) -> list[dict[str, str]]:
    """Read the frozen evaluator columns without the datasets runtime."""
    table = pq.read_table(
        parquet_path,
        columns=["task_id", "complete_prompt"],
    )
    return [
        {
            "task_id": str(row["task_id"]),
            "complete_prompt": str(row["complete_prompt"]),
        }
        for row in table.to_pylist()
    ]


def jsonl_resume_count(path: Path) -> int:
    """Count complete JSONL records before resuming an interrupted run."""
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            json.loads(line)
            count += 1
    return count


def resolve_bigcodebench_data_root() -> Path:
    paths = BenchmarkWorkerPaths.from_environment()
    if paths.bigcodebench_data_root is not None:
        return paths.bigcodebench_data_root
    snapshots = paths.hf_hub_root / "datasets--bigcode--bigcodebench" / "snapshots"
    candidates = sorted((path for path in snapshots.glob("*") if path.is_dir()), reverse=True)
    for candidate in candidates:
        if bigcodebench_parquet_path(candidate).is_file():
            return candidate
    raise FileNotFoundError(f"BigCodeBench v0.1.4 parquet is unavailable under {snapshots}")


def cruxeval_data_path() -> Path:
    return (
        BenchmarkWorkerPaths.from_environment().third_party_root
        / "cruxeval"
        / "data"
        / "cruxeval.jsonl"
    )


def ds1000_data_path() -> Path:
    return (
        BenchmarkWorkerPaths.from_environment().third_party_root
        / "DS-1000"
        / "data"
        / "ds1000.jsonl.gz"
    )


def output_path(
    run_root: Path,
    suite: str,
    arm: str,
    seed: int | None,
    crux_prompt_style: CruxPromptStyle = "canonical_direct",
) -> Path:
    suffix = "base" if seed is None else f"seed{seed}"
    extension = ".json" if suite.startswith("cruxeval_") else ".jsonl"
    if suite.startswith("cruxeval_") and crux_prompt_style != "canonical_direct":
        suffix = f"{suffix}_{crux_prompt_style.replace('_', '-')}"
    return run_root / SAMPLE_DIRECTORY / f"{suite}_{arm}_{suffix}{extension}"


def _add_cruxeval_source() -> None:
    path = str(BenchmarkWorkerPaths.from_environment().third_party_root / "cruxeval")
    if path not in sys.path:
        sys.path.insert(0, path)


def postprocess_cruxeval(
    text: str,
    mode: CruxMode,
    prompt_style: CruxPromptStyle = "canonical_direct",
) -> str:
    content = text.strip()
    if "[ANSWER]" in content:
        content = content.split("[ANSWER]", 1)[1]
    if "[/ANSWER]" in content:
        content = content.split("[/ANSWER]", 1)[0]
    content = content.strip()
    if "assert f" in content:
        content = "f" + content.split("assert f", 1)[1].strip()
    if mode == "input":
        content = content.split("==", 1)[0].strip()
        if prompt_style == "answer_prefix_v1" and not content.startswith("f("):
            content = f"f({content}"
            if not content.endswith(")"):
                content += ")"
        return content
    if "==" in content:
        return content.split("==", 1)[1].strip()
    return content


def build_cruxeval_prompt(
    item: dict[str, Any],
    mode: CruxMode,
    prompt_style: CruxPromptStyle,
) -> str:
    _add_cruxeval_source()
    from prompts import make_direct_input_prompt, make_direct_output_prompt

    prompt = (
        make_direct_input_prompt((item["code"], item["output"]))
        if mode == "input"
        else make_direct_output_prompt((item["code"], item["input"]))
    )
    if prompt_style == "canonical_direct":
        return prompt
    if prompt_style != "answer_prefix_v1":
        raise ValueError(f"Unsupported CRUXEval prompt style: {prompt_style}")
    if mode == "input":
        return prompt + "assert f("
    return prompt + f"assert f({item['input']}) == "


def batches(values: Sequence[RecordT], batch_size: int) -> Iterable[Sequence[RecordT]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def generate_texts(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_new_tokens: int,
    max_batch_context_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )
    token_envelope = len(prompts) * (
        int(encoded["input_ids"].shape[1]) + max_new_tokens
    )
    if len(prompts) > 1 and token_envelope > max_batch_context_tokens:
        midpoint = len(prompts) // 2
        return generate_texts(
            model,
            tokenizer,
            prompts[:midpoint],
            max_new_tokens,
            max_batch_context_tokens,
        ) + generate_texts(
            model,
            tokenizer,
            prompts[midpoint:],
            max_new_tokens,
            max_batch_context_tokens,
        )
    encoded = encoded.to(0)
    generated = model.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    width = encoded["input_ids"].shape[1]
    return [
        tokenizer.decode(sequence[width:], skip_special_tokens=True)
        for sequence in generated
    ]


def bigcodebench_records(
    model: Any,
    tokenizer: Any,
    problems: Sequence[dict[str, str]],
    max_new_tokens: int,
    batch_size: int,
    max_batch_context_tokens: int,
) -> Iterable[dict[str, str]]:
    for batch in batches(problems, batch_size):
        completions = generate_texts(
            model,
            tokenizer,
            [str(problem["complete_prompt"]) for problem in batch],
            max_new_tokens,
            max_batch_context_tokens,
        )
        for problem, generated in zip(batch, completions, strict=True):
            task_id = str(problem["task_id"])
            yield {
                "task_id": task_id,
                "completion": trim_completion(generated),
                "_identifier": task_id,
            }


def cruxeval_records(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    mode: CruxMode,
    batch_size: int,
    max_batch_context_tokens: int,
    prompt_style: CruxPromptStyle = "canonical_direct",
) -> dict[str, list[str]]:
    prompts: list[tuple[str, str]] = []
    with cruxeval_data_path().open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            item = json.loads(line)
            prompt = build_cruxeval_prompt(item, mode, prompt_style)
            prompts.append((f"sample_{index}", prompt))
    records: dict[str, list[str]] = {}
    for batch in batches(prompts, batch_size):
        generated = generate_texts(
            model,
            tokenizer,
            [prompt for _, prompt in batch],
            max_new_tokens,
            max_batch_context_tokens,
        )
        for (sample_id, _), text in zip(batch, generated, strict=True):
            records[sample_id] = [postprocess_cruxeval(text, mode, prompt_style)]
    return records


def ds1000_records(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    batch_size: int,
    max_batch_context_tokens: int,
) -> Iterable[dict[str, str]]:
    prompts: list[str] = []
    with gzip.open(ds1000_data_path(), "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            prompts.append(str(item["prompt"]))
    for batch in batches(prompts, batch_size):
        generated = generate_texts(
            model,
            tokenizer,
            batch,
            max_new_tokens,
            max_batch_context_tokens,
        )
        for text in generated:
            yield {"code": trim_completion(text)}


def preflight_suite(suite: str) -> None:
    required = {
        "bigcodebench": (bigcodebench_parquet_path(resolve_bigcodebench_data_root()),),
        "cruxeval_input": (cruxeval_data_path(),),
        "cruxeval_output": (cruxeval_data_path(),),
        "ds1000": (ds1000_data_path(),),
    }[suite]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{suite} preflight failed; missing required files: {missing}")


def write_records(
    suite: str,
    output: Path,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    batch_size: int,
    max_batch_context_tokens: int,
    bigcodebench_problems: Sequence[dict[str, str]],
    crux_prompt_style: CruxPromptStyle = "canonical_direct",
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if suite == "cruxeval_input":
        temporary.write_text(
            json.dumps(
                cruxeval_records(
                    model,
                    tokenizer,
                    min(max_new_tokens, 256),
                    "input",
                    batch_size,
                    max_batch_context_tokens,
                    crux_prompt_style,
                )
            ),
            encoding="utf-8",
        )
    elif suite == "cruxeval_output":
        temporary.write_text(
            json.dumps(
                cruxeval_records(
                    model,
                    tokenizer,
                    min(max_new_tokens, 256),
                    "output",
                    batch_size,
                    max_batch_context_tokens,
                    crux_prompt_style,
                )
            ),
            encoding="utf-8",
        )
    else:
        if suite == "bigcodebench":
            completed = jsonl_resume_count(temporary)
            records = bigcodebench_records(
                model,
                tokenizer,
                bigcodebench_problems[completed:],
                max_new_tokens,
                batch_size,
                max_batch_context_tokens,
            )
            mode = "a" if completed else "w"
        else:
            records = ds1000_records(
                model,
                tokenizer,
                max_new_tokens,
                batch_size,
                max_batch_context_tokens,
            )
            mode = "w"
        with temporary.open(mode, encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    os.replace(temporary, output)


@torch.no_grad()
def generate_suites(
    suites: tuple[str, ...],
    arm: str,
    seed: int | None,
    max_new_tokens: int,
    batch_size: int,
    max_batch_context_tokens: int,
    protocol_path: Path,
    input_report_path: Path,
    crux_prompt_style: CruxPromptStyle = "canonical_direct",
) -> list[Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("Official-suite generation requires CUDA")
    for suite in suites:
        preflight_suite(suite)
    protocol = load_json(protocol_path)
    input_report = load_json(input_report_path)
    resolve_model_run(protocol, input_report, arm, seed)
    bigcodebench_problems = (
        load_bigcodebench_problems(
            bigcodebench_parquet_path(resolve_bigcodebench_data_root())
        )
        if "bigcodebench" in suites
        else ()
    )
    model, tokenizer = load_model(protocol, input_report, arm, seed)
    samples_root = benchmark_root(protocol) / "samples"
    outputs: list[Path] = []
    for suite in suites:
        output = output_path(
            samples_root, suite, arm, seed, crux_prompt_style
        )
        if output.is_file():
            outputs.append(output)
            continue
        write_records(
            suite,
            output,
            model,
            tokenizer,
            max_new_tokens,
            batch_size,
            max_batch_context_tokens,
            bigcodebench_problems,
            crux_prompt_style,
        )
        outputs.append(output)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate official evaluator inputs.")
    parser.add_argument("--suite", required=True, choices=(*SUITES, "all"))
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-context-tokens", type=int, default=4096)
    parser.add_argument(
        "--crux-prompt-style",
        choices=("canonical_direct", "answer_prefix_v1"),
        default="canonical_direct",
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--input-report", type=Path, default=INPUT_REPORT_PATH)
    args = parser.parse_args()
    if args.arm == "base_no_update" and args.seed is not None:
        parser.error("base_no_update does not accept --seed")
    if args.arm != "base_no_update" and args.seed is None:
        parser.error("adapter arms require --seed")
    suites = SUITES if args.suite == "all" else (args.suite,)
    for output in generate_suites(
        suites,
        args.arm,
        args.seed,
        args.max_new_tokens,
        args.batch_size,
        args.max_batch_context_tokens,
        args.protocol,
        args.input_report,
        args.crux_prompt_style,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
