from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module():
    path = ROOT / "236_prepare_code_5m_stage0_input.py"
    spec = importlib.util.spec_from_file_location("code_5m_stage0_input", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_selection_is_deterministic_and_respects_record_boundary() -> None:
    module = _module()
    rows = [
        {"record_id": "ref-c", "text": "c"},
        {"record_id": "ref-a", "text": "a"},
        {"record_id": "ref-b", "text": "b"},
    ]
    counts = {"ref-a": 5, "ref-b": 7, "ref-c": 11}

    selected = module.select_reference_rows(rows, 12, lambda row: counts[row["record_id"]])
    repeated = module.select_reference_rows(rows, 12, lambda row: counts[row["record_id"]])

    assert [row["record_id"] for row in selected] == [row["record_id"] for row in repeated]
    assert sum(counts[row["record_id"]] for row in selected) >= 12
    assert len(selected) < len(rows)


def test_stage0_candidate_preserves_audit_tier_outside_selector_fields() -> None:
    module = _module()
    source = {
        "record_id": "raw-1",
        "text": "def square(x):\n    return x * x\n",
        "repository_or_origin": "owner/repo",
        "path": "src/math.py",
        "license": ["MIT"],
        "source_dataset": "bigcode/the-stack-dedup",
        "collected_at": "2026-07-19T00:00:00+00:00",
        "content_sha256": "hash",
    }

    candidate = module.stage0_candidate(source, "raw_like")

    assert candidate["partition"]["source_tier"] == "raw_like"
    assert candidate["partition"]["source_dataset"] == "bigcode/the-stack-dedup"
    assert "source_tier" not in module.SELECTOR_VISIBLE_PARTITION_FIELDS
    assert candidate["rights"] == {"status": "allowed", "license": "MIT"}


if __name__ == "__main__":
    test_reference_selection_is_deterministic_and_respects_record_boundary()
    test_stage0_candidate_preserves_audit_tier_outside_selector_fields()
    print("[code-5m-stage0-input] deterministic quota and selector blinding: pass")
