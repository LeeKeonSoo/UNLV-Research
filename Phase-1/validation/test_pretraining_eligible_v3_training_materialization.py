#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_pretraining_eligible_v3_training_inputs import materialize_arm


class FakeEncoded:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, *, add_special_tokens: bool) -> FakeEncoded:
        assert add_special_tokens is False
        return FakeEncoded([len(text)])


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.jsonl"
        source.write_text('{"record_id":"a","text":"aa"}\n{"record_id":"b","text":"bbb"}\n', encoding="utf-8")
        report = materialize_arm(
            arm="fixture",
            source=source,
            source_stage="fixture stage",
            output_root=root / "out",
            tokenizer=FakeTokenizer(),
            sequence_length=2,
            gradient_accumulation_steps=2,
        )
        assert report["stream_tokens"] == 4
        assert report["blocks"] == 2
        assert report["materialized_tokens"] == 4
        assert report["dropped_tail_tokens"] == 0
        assert report["optimizer_steps"] == 1
    print("[pretraining-eligible-v3-materialization] natural packing fixture: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
