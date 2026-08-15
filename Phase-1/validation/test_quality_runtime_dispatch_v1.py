from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_runtime_dispatch import (
    DistilledRuntimeConfig,
    QualityRuntimeRequest,
    TeacherOracleConfig,
    score_quality_runtime,
)
from quality_teacher_panel import PanelDecision
from quality_teacher_runtime import PanelPolicyResult


def _result(source: str, reason: str) -> tuple[PanelPolicyResult, ...]:
    return (
        PanelPolicyResult(
            policy_id="q3_substantive_payload",
            decision=PanelDecision.ABSTAIN if "retain" in reason else PanelDecision.PASS,
            first_pass=(),
            second_pass=None,
            decision_source=source,
            reason_codes=(reason,),
        ),
    )


def test_oracle_fallback_receives_only_uncertain_student_rows() -> None:
    rows = ({"chunk_uid": "certain"}, {"chunk_uid": "uncertain"})
    reviewed: list[str] = []

    def distilled_scorer(rows, *, embedding_manifest_path, ranker_manifest_path):
        assert len(rows) == 2
        return {
            "certain": _result("distilled_ranker", "quality_ranker_pass"),
            "uncertain": _result("distilled_ranker", "quality_ranker_ood_abstain"),
        }, {"runtime_method": "distilled_quality_ranker_v1", "teacher_requests": 0}

    def teacher_scorer(rows, **kwargs):
        reviewed.extend(str(row["chunk_uid"]) for row in rows)
        return {"uncertain": _result("teacher_panel", "teacher_review_pass")}, {"oracle": True}

    results, audit = score_quality_runtime(
        QualityRuntimeRequest(
            rows=rows,
            distilled=DistilledRuntimeConfig(
                embedding_manifest_path=Path("embedding.json"),
                ranker_manifest_path=Path("ranker.json"),
                oracle_fallback_enabled=True,
                maximum_oracle_fraction=0.60,
            ),
            teacher=TeacherOracleConfig(
                panel_path=Path("panel.json"),
                dotenv_path=Path(".env"),
                cache_path=Path("cache.jsonl"),
                task_workers=1,
            ),
        ),
        distilled_scorer=distilled_scorer,
        teacher_scorer=teacher_scorer,
    )

    assert reviewed == ["uncertain"]
    assert results["uncertain"][0].decision_source == "distilled_ranker"
    assert results["uncertain"][0].reason_codes == ("quality_ranker_ood_abstain",)
    assert audit["teacher_reviewed_chunks"] == 1
    assert audit["fallback_status"] == "completed_audit_only"
    assert audit["teacher_review_affects_membership"] is False


def test_large_uncertain_queue_is_retained_without_partial_order_bias() -> None:
    rows = ({"chunk_uid": "one"}, {"chunk_uid": "two"})

    def distilled_scorer(rows, **kwargs):
        return {
            str(row["chunk_uid"]): _result("distilled_ranker", "quality_ranker_ood_abstain")
            for row in rows
        }, {"runtime_method": "distilled_quality_ranker_v1", "teacher_requests": 0}

    def forbidden_teacher(rows, **kwargs):
        raise AssertionError("Oversized uncertain queues must not be partially reviewed")

    _, audit = score_quality_runtime(
        QualityRuntimeRequest(
            rows=rows,
            distilled=DistilledRuntimeConfig(
                embedding_manifest_path=Path("embedding.json"),
                ranker_manifest_path=Path("ranker.json"),
                oracle_fallback_enabled=True,
                maximum_oracle_fraction=0.10,
            ),
            teacher=TeacherOracleConfig(
                panel_path=Path("panel.json"),
                dotenv_path=Path(".env"),
                cache_path=Path("cache.jsonl"),
                task_workers=1,
            ),
        ),
        distilled_scorer=distilled_scorer,
        teacher_scorer=forbidden_teacher,
    )

    assert audit["fallback_status"] == "deferred_recalibration_required"
    assert audit["teacher_reviewed_chunks"] == 0


if __name__ == "__main__":
    test_oracle_fallback_receives_only_uncertain_student_rows()
    test_large_uncertain_queue_is_retained_without_partial_order_bias()
    print("[quality-runtime-dispatch-v1] uncertain-only oracle fallback: pass")
