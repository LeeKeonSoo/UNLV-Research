#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import PanelDecision
from quality_teacher_runtime import PanelPolicyResult


def _load_module() -> object:
    path = ROOT / "run_curation.py"
    spec = importlib.util.spec_from_file_location("run_curation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _retain_quality_scorer(rows, **_kwargs):
    return {
        str(row["chunk_uid"]): (
            PanelPolicyResult(
                policy_id="q3_substantive_payload",
                decision=PanelDecision.PASS,
                first_pass=(),
                second_pass=None,
                decision_source="declared_verifier",
                reason_codes=("source_contract_fixture_pass",),
            ),
        )
        for row in rows
    }, {"fixture": True, "input_chunks": len(rows)}


def _source(path: Path, name: str) -> dict[str, object]:
    return {
        "path": str(path),
        "text_fields": ["body"],
        "defaults": {
            "source_name": name,
            "source_uri": f"https://example.invalid/{name}",
            "collected_at": "2026-07-22T00:00:00Z",
            "pii_context": "general",
            "rights": {"status": "allowed", "license": "fixture-only"},
        },
    }


def main() -> int:
    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        valid_path = work_dir / "valid.jsonl"
        pii_path = work_dir / "pii.jsonl"
        restricted_path = work_dir / "restricted.jsonl"
        raw_allowed_path = work_dir / "raw-allowed.jsonl"
        output_dir = work_dir / "output"
        config_path = work_dir / "contract.json"
        _write_jsonl(
            valid_path,
            {
                "id": "valid",
                "body": "def stable_total(values):\n    return sum(values)\n\nThis function returns a deterministic total and has no side effects.",
                "source_name": "untrusted-row-name",
            },
        )
        _write_jsonl(
            pii_path,
            {
                "id": "pii",
                "body": "Contact the owner at 212 555 0199 before using this sufficiently long candidate record for model training.",
                "pii_context": "repository_code",
            },
        )
        _write_jsonl(
            restricted_path,
            {
                "id": "restricted",
                "body": "This sufficiently long record has a source-level restriction that must not be relaxed by the source default.",
                "rights": {"status": "restricted", "license": "restricted-fixture"},
            },
        )
        _write_jsonl(
            raw_allowed_path,
            {
                "id": "raw-allowed",
                "body": "This source declares allowed rights on the raw record and remains sufficiently long for release eligibility.",
                "rights": {"status": "allowed", "license": "raw-fixture"},
            },
        )
        source_without_rights = _source(raw_allowed_path, "raw-allowed-source")
        source_without_rights["defaults"].pop("rights")
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "mode": "normal",
                    "execution_scope": "development",
                    "input": {
                        "sources": [
                            _source(valid_path, "declared-code-source"),
                            _source(pii_path, "declared-pii-source"),
                            _source(restricted_path, "declared-restricted-source"),
                            source_without_rights,
                        ]
                    },
                    "output_dir": str(output_dir),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {"minimum_residual_chars": 40, "no_binding_budget_action": "retain_all"},
                    "claim_boundary": "fixture-only",
                }
            ),
            encoding="utf-8",
        )

        report = _load_module().materialize(
            config_path, quality_scorer=_retain_quality_scorer
        )
        release = [
            json.loads(line)
            for line in (output_dir / "stage_a_release_candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        assert report["summary"]["input_records"] == 4
        assert report["summary"]["stage_a_release_records"] == 4
        assert report["summary"]["stage_a_quarantined_records"] == 0
        by_id = {row["record_id"]: row for row in release}
        assert by_id["valid"]["provenance"]["source_name"] == "declared-code-source"
        assert by_id["raw-allowed"]["provenance"]["source_name"] == "raw-allowed-source"
        assert by_id["pii"]["hazards"]["pii_detected"] is True
        assert by_id["pii"]["hazards"]["diagnostics"]["audit_only"] is True
        assert by_id["restricted"]["rights"]["status"] == "restricted"

    print("[source-contract] source and rights metadata remain audit-only in Normal: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
