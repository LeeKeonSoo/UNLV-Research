from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _module():
    path = ROOT / "237_run_code_5m_stages.py"
    spec = importlib.util.spec_from_file_location("code_5m_stages", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(record_id: str, text: str) -> dict:
    return {
        "record_id": record_id,
        "text": text,
        "partition": {
            "split": "train",
            "bundle_id": "owner/repo",
            "repository_identity": "owner/repo",
            "path": "src/example.py",
            "change_type": "snapshot",
            "content_type": "code",
            "source_tier": "raw_like",
            "source_dataset": "bigcode/the-stack-dedup",
        },
    }


def test_selector_input_excludes_audit_source_fields() -> None:
    module = _module()
    rows = [_record("record-1", "def square(x):\n    return x * x\n")]

    selector_rows = module.selector_inputs(rows)

    assert selector_rows[0]["record_id"] == "record-1"
    assert "source_tier" not in selector_rows[0]
    assert "source_dataset" not in selector_rows[0]
    assert "utility" not in str(selector_rows[0]).lower()


def test_run_writes_disjoint_selected_and_baseline(tmp_path: Path) -> None:
    module = _module()
    input_path = tmp_path / "release_candidates.jsonl"
    policy_path = ROOT / "configs" / "temporal_code_curation_protocol_v1.json"
    rows = [
        _record("record-1", "def square(x):\n    return x * x\n"),
        _record("record-2", "def cube(x):\n    return x * x * x\n"),
        _record("record-3", "def add(x, y):\n    return x + y\n"),
    ]
    input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = module.run(input_path, policy_path, tmp_path / "stages")

    assert report["status"] == "code_5m_stages_materialized"
    assert report["stage_b_blinding_audit"]["forbidden_key_seen"] is False
    assert report["summary"]["selected_baseline_overlap_count"] == 0


if __name__ == "__main__":
    import tempfile

    test_selector_input_excludes_audit_source_fields()
    with tempfile.TemporaryDirectory() as directory:
        test_run_writes_disjoint_selected_and_baseline(Path(directory))
    print("[code-5m-stages] Stage-A/B isolation and baseline disjointness: pass")
