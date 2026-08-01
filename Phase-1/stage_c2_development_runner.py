from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from curation_artifacts import save_json, sha256_file
from reason_code_audit import build_reason_code_impact_audit
from stage_c2_model_relative_selector import select_model_relative_candidates


JsonMap = dict[str, Any]


def _read_jsonl(path: Path) -> list[JsonMap]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _evidence_by_chunk(path: Path) -> dict[str, JsonMap]:
    evidence: dict[str, JsonMap] = {}
    for row in _read_jsonl(path):
        chunk_uid = row.get("chunk_uid")
        if not isinstance(chunk_uid, str) or not chunk_uid:
            raise RuntimeError("Frozen proxy evidence requires a non-empty chunk_uid")
        if chunk_uid in evidence:
            raise RuntimeError(f"Frozen proxy evidence contains duplicate chunk_uid: {chunk_uid}")
        evidence[chunk_uid] = {key: value for key, value in row.items() if key != "chunk_uid"}
    return evidence


def _frozen_proxy_manifest(path: Path) -> JsonMap:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "frozen_proxy_evidence_ready":
        raise RuntimeError("Stage C-2 requires a frozen_proxy_evidence_ready manifest")
    return manifest


def materialize_stage_c2_development_candidate(
    *,
    stage_b_path: Path,
    frozen_proxy_evidence_path: Path,
    frozen_proxy_manifest_path: Path,
    output_dir: Path,
    selector_config: JsonMap,
) -> JsonMap:
    """Materialize a candidate-only Stage C-2 output without changing active curation."""
    evidence = _evidence_by_chunk(frozen_proxy_evidence_path)
    frozen_manifest = _frozen_proxy_manifest(frozen_proxy_manifest_path)
    candidate_rows: list[JsonMap] = []
    for row in _read_jsonl(stage_b_path):
        candidate = dict(row)
        chunk_uid = str(candidate["chunk_uid"])
        candidate["stage_c2_proxy_evidence"] = evidence.get(chunk_uid, {})
        candidate_rows.append(candidate)
    selected, rejected, audit = select_model_relative_candidates(candidate_rows, selector_config)
    audit["frozen_proxy_manifest_sha256"] = sha256_file(frozen_proxy_manifest_path)
    audit["frozen_proxy_model_id"] = frozen_manifest["model_id"]
    reason_audit = build_reason_code_impact_audit([], [], rejected)
    paths = {
        "candidate_input_chunks": output_dir / "stage_c2_candidate_input_chunks.jsonl",
        "curated_chunks": output_dir / "stage_c2_candidate_curated_chunks.jsonl",
        "not_selected_chunks": output_dir / "stage_c2_candidate_not_selected_chunks.jsonl",
        "selection_audit": output_dir / "stage_c2_candidate_selection_audit.json",
        "reason_code_audit": output_dir / "stage_c2_candidate_reason_code_audit.json",
    }
    _write_jsonl(paths["candidate_input_chunks"], candidate_rows)
    _write_jsonl(paths["curated_chunks"], selected)
    _write_jsonl(paths["not_selected_chunks"], rejected)
    save_json(paths["selection_audit"], audit)
    save_json(paths["reason_code_audit"], reason_audit)
    return {
        "schema_version": "stage-c2-development-run-v1",
        "status": "candidate_only_development_artifact",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "stage_c2_audit": audit,
        "artifacts": {name: str(path) for name, path in paths.items()},
    }


def materialize_stage_c2_development_matrix(
    *,
    corpora: dict[str, tuple[Path, Path, Path]],
    output_dir: Path,
    selector_config: JsonMap,
) -> JsonMap:
    """Materialize the fixed code, math, and general-text development matrix."""
    required_corpora = {"code_raw_like", "math_raw_like", "general_text_raw_like"}
    if set(corpora) != required_corpora:
        raise RuntimeError("Stage C-2 development matrix requires code_raw_like, math_raw_like, and general_text_raw_like")
    results: JsonMap = {}
    for corpus_id in sorted(corpora):
        stage_b_path, evidence_path, manifest_path = corpora[corpus_id]
        results[corpus_id] = materialize_stage_c2_development_candidate(
            stage_b_path=stage_b_path,
            frozen_proxy_evidence_path=evidence_path,
            frozen_proxy_manifest_path=manifest_path,
            output_dir=output_dir / corpus_id,
            selector_config=selector_config,
        )
    return {
        "schema_version": "stage-c2-development-matrix-v1",
        "status": "candidate_only_development_matrix_materialized",
        "runtime_authorization": "none_candidate_cannot_select_or_remove",
        "corpora": results,
    }
