from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from threading import Lock
from typing import Literal, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quality_teacher_panel import PolicyDecision, QualityPolicy, TeacherSpec, TeacherVote
from quality_teacher_runtime import EvaluationUnit


CACHE_SCHEMA = "quality-teacher-provider-batch-evidence-v1"


class CachedVote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: str = Field(min_length=1)
    decision: PolicyDecision
    reason_codes: tuple[str, ...] = Field(min_length=1)


class CachedUnitVotes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1)
    votes: tuple[CachedVote, CachedVote, CachedVote, CachedVote]


class TeacherBatchCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["quality-teacher-provider-batch-evidence-v1"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    teacher_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    pass_index: Literal[1, 2]
    policy_ids: tuple[str, str, str, str]
    unit_ids: tuple[str, ...] = Field(min_length=1, max_length=16)
    units: tuple[CachedUnitVotes, ...] = Field(min_length=1, max_length=16)


class TeacherBatchEvidenceStore:
    """Persist each valid provider batch before panel aggregation completes."""

    def __init__(self, *, root: Path, panel_sha256: str, runtime_sha256: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", panel_sha256):
            raise ValueError("panel_sha256 must be a lowercase SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{64}", runtime_sha256):
            raise ValueError("runtime_sha256 must be a lowercase SHA-256 digest")
        self._root = root
        self._panel_sha256 = panel_sha256
        self._runtime_sha256 = runtime_sha256
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0

    def _cache_key(
        self,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
    ) -> str:
        payload = {
            "panel_sha256": self._panel_sha256,
            "runtime_sha256": self._runtime_sha256,
            "teacher": teacher.model_dump(mode="json"),
            "policy_ids": [policy.policy_id for policy in policies],
            "pass_index": pass_index,
            "units": [unit.model_dump(mode="json") for unit in units],
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _path(self, teacher: TeacherSpec, pass_index: Literal[1, 2], cache_key: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", teacher.teacher_id).strip("._")
        teacher_digest = hashlib.sha256(teacher.teacher_id.encode("utf-8")).hexdigest()[:12]
        return self._root / f"{safe_id}-{teacher_digest}" / f"pass-{pass_index}" / f"{cache_key}.json"

    def get(
        self,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
    ) -> dict[str, tuple[TeacherVote, ...]] | None:
        cache_key = self._cache_key(teacher, policies, units, pass_index)
        path = self._path(teacher, pass_index, cache_key)
        with self._lock:
            if not path.is_file():
                self._misses += 1
                return None
            try:
                entry = TeacherBatchCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as error:
                raise RuntimeError(f"Invalid teacher batch evidence cache: {path}") from error
            self._validate_entry(entry, teacher, policies, units, pass_index, cache_key, path)
            self._hits += 1
        return {
            unit.unit_id: tuple(
                TeacherVote(
                    teacher_id=teacher.teacher_id,
                    policy_id=vote.policy_id,
                    decision=vote.decision,
                    reason_codes=vote.reason_codes,
                )
                for vote in unit.votes
            )
            for unit in entry.units
        }

    def put(
        self,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
        votes_by_unit: Mapping[str, tuple[TeacherVote, ...]],
    ) -> None:
        cache_key = self._cache_key(teacher, policies, units, pass_index)
        entry = TeacherBatchCacheEntry(
            schema_version=CACHE_SCHEMA,
            cache_key=cache_key,
            panel_sha256=self._panel_sha256,
            runtime_sha256=self._runtime_sha256,
            teacher_id=teacher.teacher_id,
            model_id=teacher.model_id,
            pass_index=pass_index,
            policy_ids=tuple(policy.policy_id for policy in policies),
            unit_ids=tuple(unit.unit_id for unit in units),
            units=tuple(
                CachedUnitVotes(
                    unit_id=unit.unit_id,
                    votes=tuple(
                        CachedVote(
                            policy_id=vote.policy_id,
                            decision=vote.decision,
                            reason_codes=vote.reason_codes,
                        )
                        for vote in votes_by_unit[unit.unit_id]
                    ),
                )
                for unit in units
            ),
        )
        path = self._path(teacher, pass_index, cache_key)
        self._validate_entry(entry, teacher, policies, units, pass_index, cache_key, path)
        with self._lock:
            if path.exists():
                existing = TeacherBatchCacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
                if existing != entry:
                    raise RuntimeError(f"Conflicting teacher batch evidence cache: {path}")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                    handle.write(entry.model_dump_json())
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            self._writes += 1

    def _validate_entry(
        self,
        entry: TeacherBatchCacheEntry,
        teacher: TeacherSpec,
        policies: tuple[QualityPolicy, ...],
        units: tuple[EvaluationUnit, ...],
        pass_index: Literal[1, 2],
        cache_key: str,
        path: Path,
    ) -> None:
        expected_policy_ids = tuple(policy.policy_id for policy in policies)
        expected_unit_ids = tuple(unit.unit_id for unit in units)
        if (
            entry.cache_key != cache_key
            or entry.panel_sha256 != self._panel_sha256
            or entry.runtime_sha256 != self._runtime_sha256
            or entry.teacher_id != teacher.teacher_id
            or entry.model_id != teacher.model_id
            or entry.pass_index != pass_index
            or entry.policy_ids != expected_policy_ids
            or entry.unit_ids != expected_unit_ids
            or tuple(unit.unit_id for unit in entry.units) != expected_unit_ids
        ):
            raise RuntimeError(f"Teacher batch evidence identity mismatch: {path}")
        policy_by_id = {policy.policy_id: policy for policy in policies}
        for unit in entry.units:
            if tuple(vote.policy_id for vote in unit.votes) != expected_policy_ids:
                raise RuntimeError(f"Teacher batch evidence policy mismatch: {path}")
            for vote in unit.votes:
                allowed = set(policy_by_id[vote.policy_id].reason_codes.for_decision(vote.decision))
                if not set(vote.reason_codes) <= allowed:
                    raise RuntimeError(f"Teacher batch evidence reason-code mismatch: {path}")

    def audit(self) -> dict[str, str | int]:
        with self._lock:
            return {
                "root": str(self._root),
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
            }


__all__ = ["CACHE_SCHEMA", "TeacherBatchEvidenceStore"]
