from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from quality_ranker_runtime import score_quality_rows_distilled
from quality_teacher_materialization import score_quality_rows
from quality_teacher_runtime import PanelPolicyResult


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonRow = Mapping[str, JsonValue]
QualityResults = dict[str, tuple[PanelPolicyResult, ...]]
QualityAudit = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DistilledRuntimeConfig:
    embedding_manifest_path: Path
    ranker_manifest_path: Path
    oracle_fallback_enabled: bool
    maximum_oracle_fraction: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_oracle_fraction <= 1.0:
            raise ValueError("Maximum oracle fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TeacherOracleConfig:
    panel_path: Path
    dotenv_path: Path
    cache_path: Path
    task_workers: int

    def __post_init__(self) -> None:
        if self.task_workers < 1:
            raise ValueError("Teacher oracle workers must be positive")


@dataclass(frozen=True, slots=True)
class QualityRuntimeRequest:
    rows: Sequence[JsonRow]
    distilled: DistilledRuntimeConfig
    teacher: TeacherOracleConfig


DistilledScorer = Callable[..., tuple[QualityResults, QualityAudit]]
TeacherScorer = Callable[..., tuple[QualityResults, QualityAudit]]


def _uid(row: JsonRow) -> str:
    value = row.get("chunk_uid") or row.get("uid")
    if not isinstance(value, str) or not value:
        raise ValueError("Quality runtime rows require chunk_uid or uid")
    return value


def _requires_oracle(results: tuple[PanelPolicyResult, ...]) -> bool:
    review_reasons = {
        "quality_ranker_ood_abstain",
        "quality_ranker_low_confidence_abstain",
    }
    return any(review_reasons & set(result.reason_codes) for result in results)


def score_quality_runtime(
    request: QualityRuntimeRequest,
    *,
    distilled_scorer: DistilledScorer = score_quality_rows_distilled,
    teacher_scorer: TeacherScorer = score_quality_rows,
) -> tuple[QualityResults, QualityAudit]:
    results, student_audit = distilled_scorer(
        request.rows,
        embedding_manifest_path=request.distilled.embedding_manifest_path,
        ranker_manifest_path=request.distilled.ranker_manifest_path,
    )
    uncertain_uids = tuple(
        uid for uid, policy_results in results.items() if _requires_oracle(policy_results)
    )
    uncertain_fraction = len(uncertain_uids) / len(request.rows) if request.rows else 0.0
    audit: QualityAudit = {
        **student_audit,
        "oracle_fallback_enabled": request.distilled.oracle_fallback_enabled,
        "maximum_oracle_fraction": request.distilled.maximum_oracle_fraction,
        "uncertain_chunks": len(uncertain_uids),
        "uncertain_fraction": uncertain_fraction,
        "teacher_reviewed_chunks": 0,
        "teacher_review_affects_membership": False,
        "fallback_status": "disabled_not_select",
    }
    if not request.distilled.oracle_fallback_enabled or not uncertain_uids:
        audit["fallback_status"] = (
            "not_required" if not uncertain_uids else "disabled_not_select"
        )
        return results, audit
    if uncertain_fraction > request.distilled.maximum_oracle_fraction:
        audit["fallback_status"] = "deferred_recalibration_required"
        return results, audit
    uncertain_set = set(uncertain_uids)
    review_rows = [row for row in request.rows if _uid(row) in uncertain_set]
    reviewed, teacher_audit = teacher_scorer(
        review_rows,
        panel_path=request.teacher.panel_path,
        dotenv_path=request.teacher.dotenv_path,
        cache_path=request.teacher.cache_path,
        task_workers=request.teacher.task_workers,
        minimum_available_teachers=3,
    )
    if set(reviewed) != uncertain_set:
        raise RuntimeError("Teacher fallback must cover every uncertain chunk exactly once")
    audit["teacher_reviewed_chunks"] = len(reviewed)
    audit["fallback_status"] = "completed_audit_only"
    audit["teacher_fallback_audit"] = teacher_audit
    return results, audit
