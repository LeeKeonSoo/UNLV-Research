#!/usr/bin/env python3
"""Validate frozen evidence required before implementing a calibrated selector."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, TypedDict


JsonMap = dict[str, Any]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs" / "calibrated_selector_contract.example.json"
REQUIRED_ARTIFACTS = (
    "reference_data",
    "held_out_calibration",
    "scope_audit",
    "external_validation_plan",
)


class PreflightReport(TypedDict):
    status: str
    checked_files: list[str]
    pending_gates: list[str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hashed_artifact(artifacts: JsonMap, name: str, checked_files: list[str]) -> None:
    artifact = artifacts.get(name)
    if not isinstance(artifact, dict):
        raise RuntimeError(f"Missing selector artifact declaration: {name}")
    path = Path(str(artifact.get("path") or ""))
    expected_sha256 = str(artifact.get("sha256") or "")
    if not path.is_file():
        raise FileNotFoundError(f"Missing selector artifact: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Selector artifact fingerprint mismatch: {name}")
    checked_files.append(str(path))


def preflight(contract_path: Path = DEFAULT_CONTRACT) -> PreflightReport:
    """Check selector evidence without scoring or selecting any data."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != "calibrated-selector-contract-v1":
        raise RuntimeError("Unexpected calibrated selector contract schema.")
    if contract.get("profile_id") != "calibrated_selector_template_v1":
        raise RuntimeError("Selector contract must target calibrated_selector_template_v1.")
    boundary = contract.get("selector_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(key) is not False
        for key in ("utility_read", "benchmark_outcomes_read", "target_token_fraction_read")
    ):
        raise RuntimeError("Calibrated selector contract violates the selector boundary.")

    status = str(contract.get("status") or "")
    if status == "template_not_runnable":
        return {
            "status": "blocked_template_not_materialized",
            "checked_files": [],
            "pending_gates": [
                "freeze reference data",
                "freeze held-out calibration",
                "complete false-positive audit",
                "complete scope audit",
                "freeze external validation plan",
                "implement selector only after every evidence gate passes",
            ],
        }
    if status != "frozen_candidate":
        raise RuntimeError("Selector contract status must be template_not_runnable or frozen_candidate.")

    if not isinstance(contract.get("selection_hypothesis"), str) or not contract["selection_hypothesis"].strip():
        raise RuntimeError("Frozen selector candidate needs a declared selection hypothesis.")
    if contract.get("score_direction") != "higher_means_more_similar_to_frozen_reference_distribution":
        raise RuntimeError("Frozen selector candidate needs the declared reference-distribution score direction.")
    false_positive_audit = contract.get("false_positive_audit")
    if not isinstance(false_positive_audit, dict) or false_positive_audit.get("status") != "passed":
        raise RuntimeError("Frozen selector candidate needs a passed false-positive audit.")

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("Frozen selector candidate needs an artifacts map.")
    checked_files: list[str] = []
    for name in REQUIRED_ARTIFACTS:
        _require_hashed_artifact(artifacts, name, checked_files)
    activation = contract.get("activation")
    if not isinstance(activation, dict) or activation.get("status") != "not_implemented":
        raise RuntimeError("This preflight only supports an unimplemented frozen selector candidate.")
    return {
        "status": "ready_for_selector_implementation",
        "checked_files": checked_files,
        "pending_gates": ["selector implementation is not present"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight a calibrated-selector evidence contract.")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(json.dumps(preflight(args.contract), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
