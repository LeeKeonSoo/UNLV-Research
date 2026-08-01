#!/usr/bin/env python3
"""Validate the frozen SLM coverage-backfill confirmatory contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

from data_eval_common import iter_jsonl_records_resilient, load_json


DEFAULT_PLAN = Path(__file__).resolve().parent / "configs" / "slm_backfill_confirmatory_plan_qwen25_0p5b_fineweb.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uid(record: Dict[str, Any]) -> str:
    return str(record.get("chunk_uid") or record.get("id") or record.get("doc_id") or "")


def _uids(path: Path) -> set[str]:
    return {_uid(record) for record in iter_jsonl_records_resilient(path) if _uid(record)}


def _assert_frozen_file(payload: Dict[str, Any], label: str) -> Path:
    path = Path(str(payload.get("path") or ""))
    if not path.exists():
        raise AssertionError(f"{label}: missing file {path}")
    expected = str(payload.get("sha256") or "")
    actual = _file_sha256(path)
    if actual != expected:
        raise AssertionError(f"{label}: sha256 mismatch expected={expected} actual={actual}")
    return path


def validate(plan_path: Path) -> Dict[str, Any]:
    plan = load_json(plan_path)
    if plan.get("status") != "frozen_before_confirmatory_training_outcomes":
        raise AssertionError("Plan status is not frozen_before_confirmatory_training_outcomes")
    if ((plan.get("framework_scope") or {}).get("utility_scope")) != "Stage C only; never selector objective":
        raise AssertionError("Utility scope contract changed")
    candidate = plan.get("frozen_candidate") or {}
    if float(candidate.get("selected_core_fraction") or -1.0) != 0.5:
        raise AssertionError("Frozen candidate is not the 50/50 selected-core mixture")
    if list(plan.get("confirmatory_seeds") or []) != [20260609, 20260610]:
        raise AssertionError("Fresh confirmatory seed contract changed")
    excluded_seed = ((plan.get("excluded_from_confirmatory_success_count") or {}).get("seed"))
    if int(excluded_seed or 0) != 20260608:
        raise AssertionError("Exploratory seed exclusion contract changed")

    candidate_path = _assert_frozen_file(candidate.get("file") or {}, "candidate")
    candidate_manifest_path = _assert_frozen_file(candidate.get("manifest") or {}, "candidate manifest")
    primary_comparator_path = _assert_frozen_file(
        ((plan.get("comparators") or {}).get("primary") or {}).get("file") or {},
        "primary comparator",
    )
    selected_path = _assert_frozen_file(
        ((plan.get("comparators") or {}).get("mechanism_diagnostic") or {}).get("file") or {},
        "selected-only mechanism diagnostic",
    )
    primary_eval_path = _assert_frozen_file(
        ((plan.get("evaluation") or {}).get("primary") or {}).get("file") or {},
        "primary eval",
    )
    secondary_eval_path = _assert_frozen_file(
        ((plan.get("evaluation") or {}).get("secondary") or {}).get("file") or {},
        "secondary eval",
    )
    holdout_manifest_path = _assert_frozen_file(
        ((plan.get("evaluation") or {}).get("holdout_manifest") or {}),
        "holdout manifest",
    )
    candidate_manifest = load_json(candidate_manifest_path)
    if candidate_manifest.get("arm_name") != candidate.get("arm"):
        raise AssertionError("Candidate arm name differs from its frozen manifest")
    holdout_manifest = load_json(holdout_manifest_path)
    if not bool((holdout_manifest.get("disjointness") or {}).get("exact_uid_disjoint")):
        raise AssertionError("Holdout manifest no longer reports exact UID disjointness")

    train_uids = _uids(candidate_path) | _uids(primary_comparator_path) | _uids(selected_path)
    primary_eval_uids = _uids(primary_eval_path)
    secondary_eval_uids = _uids(secondary_eval_path)
    overlaps = {
        "train_vs_primary_eval": len(train_uids & primary_eval_uids),
        "train_vs_secondary_eval": len(train_uids & secondary_eval_uids),
        "primary_vs_secondary_eval": len(primary_eval_uids & secondary_eval_uids),
    }
    if any(overlaps.values()):
        raise AssertionError(f"Exact UID disjointness failed: {overlaps}")
    return {
        "status": "pass",
        "plan": str(plan_path),
        "confirmatory_seeds": plan["confirmatory_seeds"],
        "candidate_arm": candidate["arm"],
        "primary_eval": str(primary_eval_path),
        "secondary_eval": str(secondary_eval_path),
        "exact_uid_overlaps": overlaps,
        "validated_frozen_files": 7,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen SLM confirmatory contract.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    result = validate(args.plan)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
