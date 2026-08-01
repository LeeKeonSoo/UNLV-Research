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


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _curation_report(audit_path: Path, policy_digest: str) -> dict[str, object]:
    return {
        "pretraining_audit": {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
            "status": "benchmark_exclusion_complete",
            "pretraining_eligible": True,
        },
        "policy_fingerprint": {
            "policy_configs": {"configs/policy_cards.json": policy_digest},
            "runtime_modules": {"run_curation.py": policy_digest},
        },
    }


def main() -> int:
    from external_evaluation.validation_integrity import build_validation_integrity_report

    with TemporaryDirectory() as directory:
        work_dir = Path(directory)
        development_candidates = work_dir / "development.jsonl"
        confirmatory_candidates = work_dir / "confirmatory.jsonl"
        development_audit = work_dir / "development_audit.json"
        confirmatory_audit = work_dir / "confirmatory_audit.json"
        development_report = work_dir / "development_report.json"
        confirmatory_report = work_dir / "confirmatory_report.json"
        _write_jsonl(development_candidates, [{"record_id": "dev-1", "text": "development-only corpus record"}])
        _write_jsonl(confirmatory_candidates, [{"record_id": "confirm-1", "text": "confirmatory-only corpus record"}])
        _write_json(
            development_audit,
            {
                "status": "benchmark_exclusion_complete",
                "pretraining_eligible": True,
                "audited_output": {"path": str(development_candidates), "sha256": _sha256(development_candidates)},
            },
        )
        _write_json(
            confirmatory_audit,
            {
                "status": "benchmark_exclusion_complete",
                "pretraining_eligible": True,
                "audited_output": {"path": str(confirmatory_candidates), "sha256": _sha256(confirmatory_candidates)},
            },
        )
        policy_digest = "a" * 64
        _write_json(development_report, _curation_report(development_audit, policy_digest))
        _write_json(confirmatory_report, _curation_report(confirmatory_audit, policy_digest))

        ready = build_validation_integrity_report(
            development_curation_report=development_report,
            confirmatory_curation_report=confirmatory_report,
        )
        assert ready["status"] == "confirmatory_ready"
        assert ready["corpus_disjointness"]["shared_record_id_count"] == 0
        assert ready["corpus_disjointness"]["shared_normalized_text_count"] == 0
        assert ready["policy_fingerprint_match"] is True
        assert ready["runtime_boundary"]["validation_metadata_visible_to_curation_runtime"] is False

        _write_jsonl(confirmatory_candidates, [{"record_id": "dev-1", "text": "development-only corpus record"}])
        _write_json(
            confirmatory_audit,
            {
                "status": "benchmark_exclusion_complete",
                "pretraining_eligible": True,
                "audited_output": {"path": str(confirmatory_candidates), "sha256": _sha256(confirmatory_candidates)},
            },
        )
        _write_json(confirmatory_report, _curation_report(confirmatory_audit, policy_digest))
        blocked = build_validation_integrity_report(
            development_curation_report=development_report,
            confirmatory_curation_report=confirmatory_report,
        )
        assert blocked["status"] == "confirmatory_blocked"
        assert "development_and_confirmatory_corpora_overlap" in blocked["blocking_reasons"]

    print("[external-validation-integrity] record/text-disjoint frozen-policy gate: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
