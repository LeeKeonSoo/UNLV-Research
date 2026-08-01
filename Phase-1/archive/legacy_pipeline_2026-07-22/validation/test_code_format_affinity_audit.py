from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_separates_copy_risk_from_format_shift(tmp_path: Path) -> None:
    from paper_evidence.code_format_affinity import AuditInputs, BenchmarkTask, build

    raw_path = tmp_path / "raw.jsonl"
    stage_a_path = tmp_path / "stage_a.jsonl"
    curated_path = tmp_path / "curated.jsonl"
    output_path = tmp_path / "audit.json"
    triple = '"' * 3
    copied = f"def add(a, b):\n    {triple}Return the sum.{triple}\n    return a + b\n"
    long_module = "\n".join(f"value_{index} = {index}" for index in range(400))
    concise = (
        f"def normalize(values):\n    {triple}Normalize a non-empty sequence.{triple}\n"
        "    total = sum(values)\n    assert total != 0\n"
        "    return [value / total for value in values]\n"
    )
    _write_jsonl(
        raw_path,
        [
            {"text": copied, "token_proxy_count": 20, "provenance": {"repository_identity": "r/a", "content_type": "code"}},
            {"text": long_module, "token_proxy_count": 1200, "provenance": {"repository_identity": "r/b", "content_type": "code"}},
        ],
    )
    _write_jsonl(
        stage_a_path,
        [
            {
                "text": concise,
                "repository_identity": "r/c",
                "content_type": "test",
                "stage_b_evidence": {"token_proxy_count": 40},
            }
        ],
    )
    _write_jsonl(
        curated_path,
        [{"text": concise, "token_proxy_count": 40, "provenance": {"repository_identity": "r/c", "content_type": "test"}}],
    )

    report = build(
        inputs=AuditInputs(
            raw_path=raw_path,
            stage_a_path=stage_a_path,
            curated_path=curated_path,
            output_path=output_path,
        ),
        benchmark_tasks=(BenchmarkTask(task_id="HumanEval/0", text=copied),),
    )

    assert report["contamination_screen"]["corpora"]["raw"]["exact_copy_candidate_records"] == 1
    assert report["contamination_screen"]["corpora"]["curated"]["exact_copy_candidate_records"] == 0
    raw_features = report["format_affinity"]["corpora"]["raw"]
    stage_a_features = report["format_affinity"]["corpora"]["stage_a"]
    curated_features = report["format_affinity"]["corpora"]["curated"]
    assert stage_a_features["token_proxy_count"] == 40
    assert curated_features["concise_share"] > raw_features["concise_share"]
    assert curated_features["function_with_docstring_share"] > raw_features["function_with_docstring_share"]
    assert report["interpretation"]["not_selector_tuning_permission"] is True
    assert output_path.exists()


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary_dir:
        test_audit_separates_copy_risk_from_format_shift(Path(temporary_dir))
    print("[code-format-affinity-audit] contract: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
