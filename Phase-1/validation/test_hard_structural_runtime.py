#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_curation import materialize, resolve_curation_mode


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hard_mode_is_development_only_until_ablation_promotion() -> None:
    try:
        resolve_curation_mode("hard")
    except RuntimeError as error:
        assert "development" in str(error)
    else:
        raise AssertionError("Hard mode must remain fail-closed for a production run.")

    assert resolve_curation_mode("hard", execution_scope="development") == {
        "mode": "hard",
        "profile_id": "hard_structural_v1",
        "authorization": "development_only_pending_n4_ablation",
    }


def test_hard_runtime_compacts_only_declared_spans_and_emits_audit_traces() -> None:
    repeated = "This repeated transport template documents retries, timeouts, authentication, and stable generated client behavior."
    rows = [
        {
            "id": "a",
            "text": "# SPDX-License-Identifier: Apache-2.0\n# Licensed under the Apache License\n\n"
            f"{repeated}\n\nThe implementation preserves an independently useful explanation of recovery behavior.",
        },
        {
            "id": "b",
            "text": f"{repeated}\n\nA separate payload explains token refresh and a distinct authentication failure path.",
        },
        {
            "id": "c",
            "text": "def add(left, right):\n    return left + right\n\n# SPDX-License-Identifier: Apache-2.0\n"
            "# Licensed under the Apache License\n\nprint(add(2, 3))",
        },
        {"id": "short-snippet", "text": "x = 1"},
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.jsonl"
        output_dir = root / "out"
        config_path = root / "config.json"
        _write_jsonl(input_path, rows)
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "hard",
                    "execution_scope": "development",
                    "input": {"candidate_files": [str(input_path)], "text_fields": ["text"], "defaults": {}},
                    "output_dir": str(output_dir),
                    "stage_a": {"policy": "text_only_v2"},
                    "stage_b": {"max_chunk_chars": 6000, "minimum_chunk_chars": 40},
                    "stage_c_selection": {},
                    "stage_c": {"no_binding_budget_action": "selection_without_binding_budget"},
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8",
        )

        report = materialize(config_path)
        transformations = [
            json.loads(line)
            for line in (output_dir / "stage_c_hard_transformations.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        curated = [
            json.loads(line)
            for line in (output_dir / "stage_c_curated_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    assert report["curation_mode"]["authorization"] == "development_only_pending_n4_ablation"
    assert report["summary"]["stage_c_hard_span_transformations"] == 3
    assert {item["reason_code"] for item in transformations} == {
        "inline_license_header_removed",
        "inline_license_comment_block_removed",
        "repeated_exact_template_span_removed",
    }
    assert all(item["post_token_proxy"] < item["pre_token_proxy"] for item in transformations)
    assert report["coverage_impact_audit"]["residual_payload"]["span_rewrite_active"] is True
    assert report["coverage_impact_audit"]["residual_payload"]["passed"] is True
    assert "SPDX-License-Identifier" not in next(row["text"] for row in curated if row["chunk_uid"].startswith("a::"))
    assert repeated not in next(row["text"] for row in curated if row["chunk_uid"].startswith("b::"))
    assert next(row["text"] for row in curated if row["chunk_uid"].startswith("short-snippet::")) == "x = 1"


if __name__ == "__main__":
    test_hard_mode_is_development_only_until_ablation_promotion()
    test_hard_runtime_compacts_only_declared_spans_and_emits_audit_traces()
    print("[hard-structural-runtime] development-only span compaction: pass")
