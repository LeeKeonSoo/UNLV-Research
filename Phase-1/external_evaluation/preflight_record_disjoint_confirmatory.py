#!/usr/bin/env python3
"""Preflight the frozen record/text-disjoint confirmatory training inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curation_artifacts import sha256_file


DEFAULT_PROTOCOL = ROOT / "protocols" / "code_record_disjoint_confirmatory_evaluation_protocol.json"
JsonMap = dict[str, Any]


class PreflightReport(TypedDict):
    status: str
    checked_files: list[str]
    pending_gates: list[str]


def _mapping(value: Any, name: str) -> JsonMap:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object.")
    return value


def _path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty path.")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_hashed_file(path: Path, expected_sha256: Any, checked_files: list[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required external-evaluation file is missing: {path}")
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise RuntimeError(f"External-evaluation fingerprint mismatch: {path}")
    checked_files.append(str(path))


def preflight(protocol_path: Path = DEFAULT_PROTOCOL) -> PreflightReport:
    """Verify frozen confirmatory artifacts before external continued pretraining."""
    protocol = _mapping(json.loads(protocol_path.read_text(encoding="utf-8")), "protocol")
    curation_input = _mapping(protocol.get("curation_input"), "curation_input")
    training = _mapping(protocol.get("training"), "training")
    checked_files: list[str] = []
    _require_hashed_file(
        _path(curation_input.get("curation_contract"), "curation_contract"),
        curation_input.get("curation_contract_sha256"),
        checked_files,
    )
    for key in ("curation_report", "benchmark_exclusion_audit", "integrity_report"):
        _require_hashed_file(
            _path(curation_input.get(key), key),
            curation_input.get(f"{key}_sha256"),
            checked_files,
        )
    audit = _mapping(
        json.loads(_path(curation_input.get("benchmark_exclusion_audit"), "benchmark_exclusion_audit").read_text(encoding="utf-8")),
        "benchmark_exclusion_audit",
    )
    integrity = _mapping(
        json.loads(_path(curation_input.get("integrity_report"), "integrity_report").read_text(encoding="utf-8")),
        "integrity_report",
    )
    materialized = _mapping(training.get("materialized_inputs"), "materialized_inputs")
    _require_hashed_file(
        _path(materialized.get("report"), "materialized_inputs.report"),
        materialized.get("report_sha256"),
        checked_files,
    )
    materialization_report = _mapping(
        json.loads(_path(materialized.get("report"), "materialized_inputs.report").read_text(encoding="utf-8")),
        "training_inputs_report",
    )
    snapshot_maps = _mapping(protocol.get("frozen_benchmark_snapshots"), "frozen_benchmark_snapshots")
    for benchmark_id, snapshot in snapshot_maps.items():
        snapshot_map = _mapping(snapshot, f"frozen_benchmark_snapshots.{benchmark_id}")
        _require_hashed_file(
            _path(snapshot_map.get("path"), f"snapshot {benchmark_id} path"),
            snapshot_map.get("sha256"),
            checked_files,
        )
    if audit.get("status") != "benchmark_exclusion_complete" or audit.get("pretraining_eligible") is not True:
        raise RuntimeError("Confirmatory benchmark-exclusion audit is incomplete.")
    if integrity.get("status") != "confirmatory_ready":
        raise RuntimeError("Confirmatory corpus integrity gate is not ready.")
    if materialization_report.get("status") != "tokenizer_materialization_complete":
        raise RuntimeError("Confirmatory tokenizer materialization is incomplete.")
    return {
        "status": "preflight_ready_for_external_training",
        "checked_files": checked_files,
        "pending_gates": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight frozen record/text-disjoint confirmatory external training.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(preflight(args.protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
