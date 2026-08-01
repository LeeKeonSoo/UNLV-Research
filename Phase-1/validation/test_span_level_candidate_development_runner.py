#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from span_level_candidate_development_runner import materialize_development_candidate


def test_development_runner_writes_separate_candidate_arm_with_frozen_input_manifest() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    rows = [
        {"chunk_uid": "a::0000", "stage_a_record_id": "a", "text": f"{repeated}\n\nRecord A payload explains retry behavior and timeout recovery in detail.", "token_proxy": 25},
        {"chunk_uid": "b::0000", "stage_a_record_id": "b", "text": f"{repeated}\n\nRecord B payload explains authentication behavior and token refresh in detail.", "token_proxy": 25},
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        stage_b_path = root / "stage_b_pass_chunks.jsonl"
        raw_input_path = root / "raw_input.jsonl"
        output_dir = root / "development_candidate_arm"
        stage_b_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        raw_input_path.write_text('{"text":"frozen raw input"}\n', encoding="utf-8")

        report = materialize_development_candidate(
            stage_b_path=stage_b_path,
            frozen_input_path=raw_input_path,
            output_dir=output_dir,
            stage_c_selection={},
            minimum_span_tokens=12,
            minimum_residual_tokens=8,
        )

        assert report["status"] == "development_candidate_materialization_complete_not_runtime_active"
        assert report["runtime_active"] is False
        assert report["frozen_input_snapshot"]["input_sha256"]
        assert report["frozen_input_snapshot"]["stage_b_pass_sha256"]
        assert report["candidate_impact_audit"]["chunks_transformed"] == 1
        reason_audit = report["reason_code_impact_audit"]
        assert reason_audit["stages"]["stage_c_span_transformation"]["reasons"]["repeated_exact_template_span_removed"]["chunks"] == 1
        assert (output_dir / "stage_c_candidate_preselection_chunks.jsonl").is_file()
        assert (output_dir / "stage_c_candidate_curated_chunks.jsonl").is_file()
        assert (output_dir / "stage_c_candidate_transformations.jsonl").is_file()
        assert (output_dir / "candidate_development_report.json").is_file()


def test_development_runner_materializes_matched_baseline_without_candidate_transformations() -> None:
    repeated = "This repeated boilerplate explains the stable transport contract shared by every generated client implementation."
    rows = [
        {"chunk_uid": "a::0000", "stage_a_record_id": "a", "text": f"{repeated}\n\nRecord A payload explains retry behavior and timeout recovery in detail.", "token_proxy": 25},
        {"chunk_uid": "b::0000", "stage_a_record_id": "b", "text": f"{repeated}\n\nRecord B payload explains authentication behavior and token refresh in detail.", "token_proxy": 25},
    ]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        stage_b_path = root / "stage_b_pass_chunks.jsonl"
        raw_input_path = root / "raw_input.jsonl"
        output_dir = root / "development_baseline_arm"
        stage_b_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        raw_input_path.write_text('{"text":"frozen raw input"}\n', encoding="utf-8")

        report = materialize_development_candidate(
            stage_b_path=stage_b_path,
            frozen_input_path=raw_input_path,
            output_dir=output_dir,
            stage_c_selection={},
            candidate_enabled=False,
            minimum_span_tokens=12,
            minimum_residual_tokens=8,
        )

        assert report["candidate_enabled"] is False
        assert report["summary"]["candidate_transformations"] == 0
        preselection = [json.loads(line) for line in (output_dir / "stage_c_candidate_preselection_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
        assert preselection[1]["text"] == rows[1]["text"]


if __name__ == "__main__":
    test_development_runner_writes_separate_candidate_arm_with_frozen_input_manifest()
    test_development_runner_materializes_matched_baseline_without_candidate_transformations()
    print("[span-level-candidate-development-runner] frozen candidate arm: pass")
