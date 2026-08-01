#!/usr/bin/env python3
"""Validate real-corpus Stage-0/Coverage metadata audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(script: str):
    path = ROOT / script
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _stage0_record(record_id: str, *, eligible: bool = True, reasons: list[str] | None = None) -> dict:
    return {
        "record_id": record_id,
        "schema_version": "fixture",
        "text": "def add(a, b):\n    return a + b\n",
        "rights": {"status": "allowed", "license": "MIT"},
        "release_eligibility": {"eligible": eligible},
        "quarantine": {"reasons": reasons or []},
        "hazards": {"poisoning_suspected": bool(reasons)},
        "provenance": {
            "source_name": "fixture",
            "source_uri": f"fixture://{record_id}",
            "collected_at": "2026-06-23T00:00:00Z",
            "original_sha256": "0" * 64,
            "normalized_sha256": "1" * 64,
        },
        "code_domain_v2_source_pool": "fixture_pool",
    }


def _chunk(uid: str, repo: str, content_type: str, path: str) -> dict:
    return {
        "chunk_uid": uid,
        "record_id": uid,
        "bundle_id": repo.replace("/", "__"),
        "repository_identity": repo,
        "content_type": content_type,
        "change_type": "modified",
        "path": path,
        "stage_a_pass": True,
        "text": "def add(a, b):\n    return a + b\n",
    }


def main() -> int:
    module = _load("169_build_real_corpus_stage0_coverage_audit.py")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stage0_dir = tmp_path / "stage0"
        _write_jsonl(stage0_dir / "train" / "release_candidates.jsonl", [_stage0_record("r1"), _stage0_record("r2")])
        _write_jsonl(
            stage0_dir / "train" / "quarantined_candidates.jsonl",
            [_stage0_record("q1", eligible=False, reasons=["poisoning_suspected"])],
        )
        _write_jsonl(stage0_dir / "development" / "release_candidates.jsonl", [])
        _write_jsonl(stage0_dir / "development" / "quarantined_candidates.jsonl", [])
        _write_jsonl(stage0_dir / "confirmatory" / "release_candidates.jsonl", [])
        _write_jsonl(stage0_dir / "confirmatory" / "quarantined_candidates.jsonl", [])
        stage_a = tmp_path / "stage_a.jsonl"
        selected = tmp_path / "selected.jsonl"
        a_rows = [
            _chunk("c1", "fixture/repo-a", "code", "src/a.py"),
            _chunk("c2", "fixture/repo-b", "test", "tests/test_b.py"),
            _chunk("c3", "fixture/repo-c", "documentation", "README.md"),
        ]
        selected_rows = [
            {
                "chunk_uid": row["chunk_uid"],
                "provenance": {
                    "repository_identity": row["repository_identity"],
                    "bundle_id": row["bundle_id"],
                    "content_type": row["content_type"],
                    "change_type": row["change_type"],
                    "path": row["path"],
                },
                "stage_b_evidence": {
                    "coverage_buckets": {
                        "repository_identity": row["repository_identity"],
                        "bundle_id": row["bundle_id"],
                        "content_type": row["content_type"],
                        "change_type": row["change_type"],
                        "path_family": "root" if "/" not in row["path"] else row["path"].split("/", 1)[0],
                    }
                },
            }
            for row in a_rows[:2]
        ]
        _write_jsonl(stage_a, a_rows)
        _write_jsonl(selected, selected_rows)
        report = module.build(
            stage0_dir,
            stage_a,
            selected,
            tmp_path / "real_corpus_stage0_coverage_audit.json",
            tmp_path / "real_corpus_stage0_coverage_audit.md",
        )
    assert report["status"] == "real_corpus_stage0_coverage_audit_passed_with_scope_caveats"
    assert not report["blockers"]
    assert report["stage0"]["release_candidate_count"] == 2
    assert report["stage0"]["quarantined_candidate_count"] == 1
    assert report["coverage"]["support_scope"] == "source_or_repository_bucket_fallback"
    assert report["coverage"]["true_domain_coverage_claim_allowed"] is False
    assert report["coverage"]["distribution_support"]["content_type"]["retained_bucket_ratio"] == 0.666667
    assert "true_domain_coverage_not_claimable_without_explicit_domain_metadata" in report["caveats"]
    print("[real-corpus-stage0-coverage] metadata support and claim boundary checks: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
