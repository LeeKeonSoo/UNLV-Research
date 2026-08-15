#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_evaluation.official_suite_generator import (
    batches,
    bigcodebench_parquet_path,
    build_cruxeval_prompt,
    generate_texts,
    jsonl_resume_count,
    load_bigcodebench_problems,
    output_path,
    postprocess_cruxeval,
)


class FakeBatch(dict[str, torch.Tensor]):
    def to(self, _device: int) -> "FakeBatch":
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, prompts: list[str], **_kwargs: object) -> FakeBatch:
        width = max(len(prompt) for prompt in prompts)
        return FakeBatch(input_ids=torch.zeros((len(prompts), width), dtype=torch.long))

    def decode(self, tokens: torch.Tensor, **_kwargs: object) -> str:
        return ",".join(str(int(token)) for token in tokens)


class FakeModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, **_kwargs: object) -> torch.Tensor:
        self.batch_sizes.append(int(input_ids.shape[0]))
        suffix = torch.full((input_ids.shape[0], max_new_tokens), 7, dtype=torch.long)
        return torch.cat((input_ids, suffix), dim=1)


def main() -> int:
    run_root = Path("D:/runs")
    actual = output_path(run_root, "ds1000", "hard_natural", 303)
    expected = run_root / "official_suite_samples" / "ds1000_hard_natural_seed303.jsonl"
    assert actual == expected
    assert output_path(run_root, "cruxeval_input", "base_no_update", None).suffix == ".json"
    assert output_path(
        run_root,
        "cruxeval_input",
        "base_no_update",
        None,
        "answer_prefix_v1",
    ).name == "cruxeval_input_base_no_update_base_answer-prefix-v1.json"
    assert postprocess_cruxeval("[ANSWER]\nassert f(3) == 7\n[/ANSWER]", "input") == "f(3)"
    assert postprocess_cruxeval("[ANSWER]\nassert f(3) == 7\n[/ANSWER]", "output") == "7"
    input_prompt = build_cruxeval_prompt(
        {"code": "def f(x):\n    return x + 1", "output": "7"},
        "input",
        "answer_prefix_v1",
    )
    output_prompt = build_cruxeval_prompt(
        {"code": "def f(x):\n    return x + 1", "input": "6"},
        "output",
        "answer_prefix_v1",
    )
    assert input_prompt.endswith("[ANSWER]\nassert f(")
    assert output_prompt.endswith("[ANSWER]\nassert f(6) == ")
    assert postprocess_cruxeval("3) == 7", "input", "answer_prefix_v1") == "f(3)"
    assert postprocess_cruxeval("7", "output", "answer_prefix_v1") == "7"
    fixture_root = Path("D:/fixture")
    assert bigcodebench_parquet_path(fixture_root) == fixture_root / "data" / "v0.1.4-00000-of-00001.parquet"
    with TemporaryDirectory() as directory:
        fixture_parquet = Path(directory) / "bigcodebench.parquet"
        pq.write_table(
            pa.table(
                {
                    "task_id": ["BigCodeBench/0"],
                    "complete_prompt": ["def answer():\n"],
                }
            ),
            fixture_parquet,
        )
        assert load_bigcodebench_problems(fixture_parquet) == [
            {
                "task_id": "BigCodeBench/0",
                "complete_prompt": "def answer():\n",
            }
        ]
        partial_output = Path(directory) / "partial.jsonl.tmp"
        partial_output.write_text(
            '{"task_id":"BigCodeBench/0","completion":"pass"}\n',
            encoding="utf-8",
        )
        assert jsonl_resume_count(partial_output) == 1
    assert list(batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    model = FakeModel()
    generated = generate_texts(
        model,
        FakeTokenizer(),
        ["a", "b", "c", "d"],
        max_new_tokens=2,
        max_batch_context_tokens=8,
    )
    assert model.batch_sizes == [2, 2]
    assert generated == ["7,7"] * 4
    print("[official-suite-generator] output path contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
