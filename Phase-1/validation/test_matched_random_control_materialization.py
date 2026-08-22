#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_matched_random_control import materialize


class FakeEncoded:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> FakeEncoded:
        assert add_special_tokens is False
        return FakeEncoded([ord(character) for character in text])


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "raw.jsonl"
        source.write_text(
            "".join(
                json.dumps({"record_id": f"r{index}", "text": text}) + "\n"
                for index, text in enumerate(("aa", "bbb", "c", "dddd", "ee", "fff"))
            ),
            encoding="utf-8",
        )
        target_report = root / "target.json"
        target_report.write_text(
            json.dumps({"arms": {"hard_natural": {"materialized_tokens": 8}}}),
            encoding="utf-8",
        )
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": "matched-random-control-materialization-v1",
                    "status": "frozen_before_materialization",
                    "source": {"path": str(source), "raw_materialized_tokens": 12},
                    "target": {"report": str(target_report), "arm": "hard_natural"},
                    "output_arm": "random_math_hard_matched",
                    "sampling": {"seed": 1701, "preserve_source_order": True},
                    "packing": {"sequence_length": 2, "gradient_accumulation_steps": 2},
                    "output_root": str(root / "out"),
                }
            ),
            encoding="utf-8",
        )

        first = materialize(contract, tokenizer=FakeTokenizer())
        second = materialize(contract, tokenizer=FakeTokenizer())

        assert first == second
        assert set(first["arms"]) == {"random_math_hard_matched"}
        arm = first["arms"]["random_math_hard_matched"]
        assert arm["materialized_tokens"] == 8
        assert arm["blocks"] == 4
        assert arm["optimizer_steps"] == 2
        assert 8 <= arm["stream_tokens"] < 12
        block_payload = torch.load(arm["blocks_path"], map_location="cpu", weights_only=True)
        assert tuple(block_payload["input_ids"].shape) == (4, 2)
        rows = [json.loads(line) for line in Path(arm["arm_path"]).read_text(encoding="utf-8").splitlines()]
        source_indices = [int(row["source_index"]) for row in rows]
        assert source_indices == sorted(source_indices)
        assert first["selector_boundary"]["benchmark_outcomes_read"] is False
        assert first["control_boundary"]["runtime_policy"] is False
    print("[matched-random-control] deterministic exact-size materialization: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
