from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Encoded:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> _Encoded:
        return _Encoded([ord(character) % 29 + 1 for character in text])


def _module():
    path = ROOT / "238_prepare_code_5m_external_validation.py"
    spec = importlib.util.spec_from_file_location("code_5m_external_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_materialize_keeps_stage0_raw_safe_and_stage_b_curated_distinct(tmp_path: Path) -> None:
    module = _module()
    raw_path = tmp_path / "raw_safe.jsonl"
    curated_path = tmp_path / "curated.jsonl"
    raw_path.write_text(
        "\n".join(
            json.dumps({"record_id": f"raw-{index}", "text": "r" * 90})
            for index in range(4)
        ) + "\n",
        encoding="utf-8",
    )
    curated_path.write_text(
        "\n".join(
            json.dumps({"chunk_uid": f"curated-{index}", "record_id": f"record-{index}", "text": "c" * 50})
            for index in range(4)
        ) + "\n",
        encoding="utf-8",
    )

    report = module.materialize(raw_path, curated_path, tmp_path / "out", _Tokenizer(), 16, 2)

    assert report["arms"]["raw_safe_natural"]["records"] == 4
    assert report["arms"]["curated_natural"]["records"] == 4
    assert report["arms"]["raw_safe_natural"]["optimizer_steps"] > report["arms"]["curated_natural"]["optimizer_steps"]
    assert report["arms"]["raw_safe_natural"]["source_stage"] == "Stage 0 release"
    assert report["arms"]["curated_natural"]["source_stage"] == "Stage B selected"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_materialize_keeps_stage0_raw_safe_and_stage_b_curated_distinct(Path(directory))
    print("[prepare-code-5m-external-validation] Stage-0 raw-safe and Stage-B curated arms: pass")
