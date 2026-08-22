from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from all_policy_stage_b import RedundancyPolicyResult, apply_redundancy_policy
from redundancy_equivalence import RedundancyMode, WitnessKind
from redundancy_mode_policy import (
    RedundancyFamilyProposal,
    RedundancyPlan,
    RedundancyRemovalProposal,
)
from redundancy_v2 import RedundancySettings
from redundancy_v2_retrieval import CandidatePair


JsonMap = dict[str, Any]
SCHEMA_VERSION = "redundancy-runtime-checkpoint-v2"
IMPLEMENTATION_FILES = (
    "all_policy_stage_b.py",
    "redundancy_checkpoint.py",
    "redundancy_equivalence.py",
    "redundancy_mode_policy.py",
    "redundancy_v2.py",
    "redundancy_v2_retrieval.py",
)


@dataclass(frozen=True, slots=True)
class CheckpointedRedundancyResult:
    result: RedundancyPolicyResult
    checkpoint_hit: bool
    identity_sha256: str
    checkpoint_path: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(
    rows: tuple[JsonMap, ...] | list[JsonMap],
    mode: RedundancyMode,
    settings: RedundancySettings,
) -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    header = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode.value,
        "settings": asdict(settings),
        "implementations": {
            name: _file_sha256(root / name) for name in IMPLEMENTATION_FILES
        },
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for row in rows:
        digest.update(b"\0")
        digest.update(
            json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
        )
    return digest.hexdigest()


def _plan_payload(plan: RedundancyPlan) -> JsonMap:
    return {
        "mode": plan.mode.value,
        "input_uids": list(plan.input_uids),
        "proposed_survivor_uids": list(plan.proposed_survivor_uids),
        "removals": [
            {
                **asdict(removal),
                "mode": removal.mode.value,
                "witness_kind": removal.witness_kind.value,
            }
            for removal in plan.removals
        ],
        "families": [asdict(family) for family in plan.families],
        "candidate_pairs": [asdict(pair) for pair in plan.candidate_pairs],
        "representative_selection": plan.representative_selection,
        "coverage_veto_required": plan.coverage_veto_required,
    }


def _plan_from_payload(payload: JsonMap) -> RedundancyPlan:
    return RedundancyPlan(
        mode=RedundancyMode(str(payload["mode"])),
        input_uids=tuple(str(uid) for uid in payload["input_uids"]),
        proposed_survivor_uids=tuple(
            str(uid) for uid in payload["proposed_survivor_uids"]
        ),
        removals=tuple(
            RedundancyRemovalProposal(
                removed_uid=str(item["removed_uid"]),
                representative_uid=str(item["representative_uid"]),
                mode=RedundancyMode(str(item["mode"])),
                reason_code=str(item["reason_code"]),
                witness_kind=WitnessKind(str(item["witness_kind"])),
                evidence_sha256=str(item["evidence_sha256"]),
                family_id=str(item["family_id"]),
                removed_token_count=int(item["removed_token_count"]),
                coverage_veto_required=bool(item["coverage_veto_required"]),
                benchmark_outcomes_read=bool(item["benchmark_outcomes_read"]),
                utility_read=bool(item["utility_read"]),
            )
            for item in payload["removals"]
        ),
        families=tuple(
            RedundancyFamilyProposal(
                family_id=str(item["family_id"]),
                representative_uid=str(item["representative_uid"]),
                member_uids=tuple(str(uid) for uid in item["member_uids"]),
                evidence_sha256=str(item["evidence_sha256"]),
            )
            for item in payload["families"]
        ),
        candidate_pairs=tuple(
            CandidatePair(
                left_uid=str(item["left_uid"]),
                right_uid=str(item["right_uid"]),
                retrieval_reasons=tuple(
                    str(reason) for reason in item["retrieval_reasons"]
                ),
            )
            for item in payload["candidate_pairs"]
        ),
        authority_decisions=(),
        representative_selection=str(payload["representative_selection"]),
        coverage_veto_required=bool(payload["coverage_veto_required"]),
    )


def _result_from_payload(payload: JsonMap) -> RedundancyPolicyResult:
    return RedundancyPolicyResult(
        survivors=tuple(dict(row) for row in payload["survivors"]),
        removals=tuple(dict(row) for row in payload["removed_rows"]),
        plan=_plan_from_payload(dict(payload["plan"])),
        audit=dict(payload["audit"]),
    )


def load_or_build_redundancy(
    rows: tuple[JsonMap, ...] | list[JsonMap],
    *,
    mode: RedundancyMode,
    settings: RedundancySettings,
    checkpoint_path: Path,
) -> CheckpointedRedundancyResult:
    identity = _identity(rows, mode, settings)
    if checkpoint_path.exists():
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") == SCHEMA_VERSION
            and payload.get("identity_sha256") == identity
        ):
            return CheckpointedRedundancyResult(
                result=_result_from_payload(payload),
                checkpoint_hit=True,
                identity_sha256=identity,
                checkpoint_path=str(checkpoint_path),
            )

    result = apply_redundancy_policy(rows, mode=mode, settings=settings)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity,
        "survivors": list(result.survivors),
        "removed_rows": list(result.removals),
        "plan": _plan_payload(result.plan),
        "audit": result.audit,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)
    return CheckpointedRedundancyResult(
        result=result,
        checkpoint_hit=False,
        identity_sha256=identity,
        checkpoint_path=str(checkpoint_path),
    )


__all__ = ["CheckpointedRedundancyResult", "load_or_build_redundancy"]
