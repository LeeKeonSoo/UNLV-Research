#!/usr/bin/env python3
"""Verify that the current seven-benchmark v3 external evaluation can be frozen safely."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols" / "code_7benchmark_pretraining_eligible_v3_execution.json"


class PreflightReport(TypedDict):
    status: str
    checked_files: list[str]
    pending_gates: list[str]


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, checked_files: list[str]) -> None:
    """Require a regular file and record it in the preflight result."""
    if not path.is_file():
        raise FileNotFoundError(f"Required external-evaluation file is missing: {path}")
    checked_files.append(str(path))


def require_hashed_file(path: Path, expected_sha256: str, checked_files: list[str]) -> None:
    """Require a file whose content matches the execution-contract fingerprint."""
    require_file(path, checked_files)
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Execution-contract fingerprint mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def preflight(protocol_path: Path = DEFAULT_PROTOCOL) -> PreflightReport:
    """Check immutable artifacts and report declared gates without running a model."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    curation_input = protocol["curation_input"]
    snapshots = protocol["frozen_benchmark_snapshots"]
    checked_files: list[str] = []

    require_hashed_file(
        ROOT / curation_input["curation_contract"],
        curation_input["curation_contract_sha256"],
        checked_files,
    )
    require_hashed_file(
        Path(curation_input["curation_report"]),
        curation_input["curation_report_sha256"],
        checked_files,
    )
    require_hashed_file(
        Path(curation_input["benchmark_exclusion_audit"]),
        curation_input["benchmark_exclusion_audit_sha256"],
        checked_files,
    )
    materialized_inputs = protocol["training"]["materialized_inputs"]
    require_hashed_file(
        Path(materialized_inputs["report"]),
        materialized_inputs["report_sha256"],
        checked_files,
    )
    require_file(Path(protocol["target_model"]["snapshot_path"]) / "config.json", checked_files)

    snapshot_root = Path(snapshots["directory"])
    for snapshot_name in (
        "livecodebench_code_generation_lite.json",
        "bigcodebench_complete.json",
        "cruxeval_input_prediction.json",
        "cruxeval_output_prediction.json",
        "ds1000.json",
    ):
        require_file(snapshot_root / snapshot_name, checked_files)

    third_party_root = Path("D:/UNLV-Research/third_party")
    for evaluator_file in (
        third_party_root / "CRUXEval" / "evaluation" / "evaluate_generations.py",
        third_party_root / "CRUXEval" / "prompts.py",
        third_party_root / "DS-1000" / "test_ds1000.py",
        third_party_root / "LiveCodeBench" / "lcb_runner" / "benchmarks" / "code_generation.py",
        third_party_root / "bigcodebench" / "bigcodebench",
    ):
        if evaluator_file.name == "bigcodebench":
            if not evaluator_file.is_dir():
                raise FileNotFoundError(f"Required official evaluator directory is missing: {evaluator_file}")
            checked_files.append(str(evaluator_file))
        else:
            require_file(evaluator_file, checked_files)
    if importlib.util.find_spec("evalplus") is None:
        raise ModuleNotFoundError("EvalPlus is unavailable in the active Python environment")

    pending_gates = ["v3 adapter training has not been materialized"]
    temporal = protocol["temporal_declaration"]
    if temporal["model_pretraining_cutoff"] is None:
        pending_gates.append("Qwen3-4B pretraining cutoff lacks an auditable declaration")
    if temporal["raw_corpus_snapshot_end"] is None:
        pending_gates.append("raw corpus snapshot end lacks an auditable declaration")
    return {
        "status": "preflight_ready_with_declared_blocks" if pending_gates else "ready_to_generate",
        "checked_files": checked_files,
        "pending_gates": pending_gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight the current seven-benchmark external evaluation contract.")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--require-runnable", action="store_true")
    args = parser.parse_args()
    report = preflight(args.protocol)
    print(json.dumps(report, indent=2))
    if args.require_runnable and report["pending_gates"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
