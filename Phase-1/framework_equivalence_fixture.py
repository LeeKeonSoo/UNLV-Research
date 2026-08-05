from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from quality_teacher_panel import PanelDecision, PolicyDecision, TeacherVote
from quality_teacher_runtime import PanelPolicyResult
from run_curation import materialize

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class EquivalenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


QUALITY_POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _fixture_quality_scorer(rows, **_kwargs):
    results = {}
    for row in rows:
        policy_results = []
        for policy_id in QUALITY_POLICY_IDS:
            votes = tuple(
                TeacherVote(
                    teacher_id=f"fixture-teacher-{index}",
                    policy_id=policy_id,
                    decision=PolicyDecision.PASS,
                    reason_codes=("fixture_pass",),
                )
                for index in range(3)
            )
            policy_results.append(
                PanelPolicyResult(
                    policy_id=policy_id,
                    decision=PanelDecision.PASS,
                    first_pass=votes,
                    second_pass=None,
                )
            )
        results[str(row["chunk_uid"])] = tuple(policy_results)
    return results, {"fixture": True, "input_chunks": len(rows)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: JsonValue) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def curated_projection_hash(records: tuple[EquivalenceRecord, ...]) -> str:
    with TemporaryDirectory() as directory:
        work = Path(directory)
        candidates = work / "candidates.jsonl"
        output = work / "output"
        audit = work / "benchmark_exclusion.json"
        config = work / "run.json"
        candidates.write_text(
            "".join(json.dumps(row.model_dump(), ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        audit.write_text(
            json.dumps(
                {
                    "status": "benchmark_exclusion_complete",
                    "pretraining_eligible": True,
                    "audited_output": {"path": str(candidates), "sha256": _sha256(candidates)},
                }
            ),
            encoding="utf-8",
        )
        config.write_text(
            json.dumps(
                {
                    "schema_version": "curation-run-contract-v1",
                    "status": "frozen_before_stage_a_b_c_materialization",
                    "curation_mode": "normal",
                    "execution_scope": "development",
                    "input": {"candidate_files": [str(candidates)], "text_fields": ["text"], "defaults": {}},
                    "output_dir": str(output),
                    "pretraining_audit_path": str(audit),
                    "stage_b": {"max_chunk_chars": 6000},
                    "stage_c": {
                        "minimum_residual_chars": 40,
                        "no_binding_budget_action": "selection_without_binding_budget",
                    },
                    "claim_boundary": "block-8-output-equivalence-fixture",
                }
            ),
            encoding="utf-8",
        )
        materialize(config, quality_scorer=_fixture_quality_scorer)
        rows = [
            json.loads(line)
            for line in (output / "stage_c_curated_chunks.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        projection: list[JsonValue] = [
            {
                "chunk_uid": row["chunk_uid"],
                "text": row["text"],
                "stage_b_trigger": row["stage_b_decision"]["trigger"],
                "stage_b_policy_trigger": row["stage_b_policy"]["trigger"],
                "quality_decision": row["quality_retention_decision"]["decision"],
            }
            for row in sorted(rows, key=lambda item: str(item["chunk_uid"]))
        ]
        return _canonical_sha256(projection)


__all__ = ["EquivalenceRecord", "curated_projection_hash"]
