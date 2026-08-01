from __future__ import annotations

import json
from pathlib import Path

from stage_c2_development_runner import (
    materialize_stage_c2_development_candidate,
    materialize_stage_c2_development_matrix,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_materialize_stage_c2_development_candidate_keeps_runtime_surface_unchanged(tmp_path: Path) -> None:
    # Given: a frozen Stage-B snapshot and separately frozen proxy evidence.
    stage_b_path = tmp_path / "stage_b.jsonl"
    evidence_path = tmp_path / "evidence.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(stage_b_path, [
        {"chunk_uid": "a", "stage_a_record_id": "r1", "text": "representative"},
        {"chunk_uid": "b", "stage_a_record_id": "r2", "text": "redundant"},
    ])
    _write_jsonl(evidence_path, [
        {"chunk_uid": "a", "semantic_bucket": "fixture-family", "embedding": [1.0, 0.0], "familiarity": 0.1, "novelty": 0.9, "gradient_alignment": 0.7},
        {"chunk_uid": "b", "semantic_bucket": "fixture-family", "embedding": [0.999, 0.001], "familiarity": 0.9, "novelty": 0.1, "gradient_alignment": 0.0},
    ])
    manifest_path.write_text(json.dumps({"status": "frozen_proxy_evidence_ready", "model_id": "fixture"}), encoding="utf-8")

    # When: the candidate-only development runner materializes its output.
    manifest = materialize_stage_c2_development_candidate(
        stage_b_path=stage_b_path,
        frozen_proxy_evidence_path=evidence_path,
        frozen_proxy_manifest_path=manifest_path,
        output_dir=tmp_path / "output",
        selector_config={
            "semantic_index": {"cosine_threshold": 0.98},
            "evidence_thresholds": {"minimum_familiarity": 0.8, "maximum_novelty": 0.2, "maximum_gradient_alignment": 0.05},
        },
    )

    # Then: it emits a candidate artifact and declares no runtime authority.
    assert manifest["status"] == "candidate_only_development_artifact"
    assert manifest["runtime_authorization"] == "none_candidate_cannot_select_or_remove"
    assert manifest["stage_c2_audit"]["candidate_removed_chunks"] == 1
    reason_report = json.loads(Path(manifest["artifacts"]["reason_code_audit"]).read_text(encoding="utf-8"))
    assert reason_report["stages"]["stage_c_compaction"]["reasons"]["model_relative_redundant_family_member"]["chunks"] == 1
    assert Path(manifest["artifacts"]["curated_chunks"]).is_file()


def test_materialize_stage_c2_development_matrix_requires_all_three_declared_corpora(tmp_path: Path) -> None:
    # Given: three frozen corpus snapshots with their separately frozen evidence files.
    corpora: dict[str, tuple[Path, Path, Path]] = {}
    for corpus_id in ("code_raw_like", "math_raw_like", "general_text_raw_like"):
        stage_b_path = tmp_path / f"{corpus_id}_stage_b.jsonl"
        evidence_path = tmp_path / f"{corpus_id}_evidence.jsonl"
        manifest_path = tmp_path / f"{corpus_id}_manifest.json"
        _write_jsonl(stage_b_path, [{"chunk_uid": "one", "stage_a_record_id": corpus_id, "text": corpus_id}])
        _write_jsonl(evidence_path, [{"chunk_uid": "one", "semantic_bucket": corpus_id, "embedding": [1.0], "familiarity": 0.1, "novelty": 0.9, "gradient_alignment": 0.5}])
        manifest_path.write_text(json.dumps({"status": "frozen_proxy_evidence_ready", "model_id": corpus_id}), encoding="utf-8")
        corpora[corpus_id] = (stage_b_path, evidence_path, manifest_path)

    # When: the declared development matrix is materialized.
    manifest = materialize_stage_c2_development_matrix(
        corpora=corpora,
        output_dir=tmp_path / "matrix",
        selector_config={"semantic_index": {"cosine_threshold": 0.98}, "evidence_thresholds": {}},
    )

    # Then: every required corpus receives an independent candidate-only artifact.
    assert set(manifest["corpora"]) == {"code_raw_like", "math_raw_like", "general_text_raw_like"}
    assert manifest["runtime_authorization"] == "none_candidate_cannot_select_or_remove"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temporary_directory:
        test_materialize_stage_c2_development_candidate_keeps_runtime_surface_unchanged(Path(temporary_directory))
        test_materialize_stage_c2_development_matrix_requires_all_three_declared_corpora(Path(temporary_directory))
    print("[stage-c2-development-runner] candidate-only development arm: pass")
