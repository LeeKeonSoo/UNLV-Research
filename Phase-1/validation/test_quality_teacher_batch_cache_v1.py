from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_batch_cache import TeacherBatchEvidenceStore
from quality_teacher_batch_runtime import (
    PolicySetBatchGenerationRequest,
    evaluate_quality_units_batched,
)
from quality_teacher_panel import load_teacher_panel
from quality_teacher_runtime import EvaluationUnit, TeacherGenerationUnavailable


PANEL = ROOT / "configs" / "quality_teacher_panel_v2.json"


def _response(request: PolicySetBatchGenerationRequest) -> str:
    reason_by_policy = {
        "q1_correctness_evidence": "observable_correctness_evidence",
        "q2_semantic_coherence": "recoverable_semantic_unit",
        "q3_substantive_payload": "substantive_payload_present",
        "q4_learnable_relations": "recoverable_relation_present",
    }
    return json.dumps(
        {
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "policies": [
                        {
                            "policy_id": policy.policy_id,
                            "decision": "pass",
                            "reason_codes": [reason_by_policy[policy.policy_id]],
                        }
                        for policy in request.policies
                    ],
                }
                for unit in request.units
            ]
        }
    )


class CountingAdapter:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.calls = 0
        self.unavailable = unavailable

    def generate_policy_batch(self, request: PolicySetBatchGenerationRequest) -> str:
        self.calls += 1
        if self.unavailable:
            raise TeacherGenerationUnavailable(request.teacher_id, "controlled_unavailable")
        return _response(request)


def test_completed_teacher_batches_survive_before_panel_observation() -> None:
    panel = load_teacher_panel(PANEL)
    units = (
        EvaluationUnit(
            unit_id="unit-0",
            text="substantive payload",
            declared_context=None,
            attached_evidence=(),
        ),
    )
    adapters = {teacher.teacher_id: CountingAdapter() for teacher in panel.teachers}

    with TemporaryDirectory() as directory:
        store = TeacherBatchEvidenceStore(
            root=Path(directory),
            panel_sha256="a" * 64,
            runtime_sha256="b" * 64,
        )
        first = evaluate_quality_units_batched(
            panel,
            adapters,
            units,
            evidence_store=store,
        )
        second = evaluate_quality_units_batched(
            panel,
            adapters,
            units,
            evidence_store=store,
        )

        assert first == second
        assert all(adapter.calls == 1 for adapter in adapters.values())
        assert len(tuple(Path(directory).rglob("*.json"))) == 3
        assert store.audit() == {
            "root": str(Path(directory)),
            "hits": 3,
            "misses": 3,
            "writes": 3,
        }


def test_runtime_identity_change_cannot_reuse_teacher_evidence() -> None:
    panel = load_teacher_panel(PANEL)
    unit = EvaluationUnit(
        unit_id="unit-0",
        text="substantive payload",
        declared_context=None,
        attached_evidence=(),
    )
    adapters = {teacher.teacher_id: CountingAdapter() for teacher in panel.teachers}

    with TemporaryDirectory() as directory:
        first_store = TeacherBatchEvidenceStore(
            root=Path(directory),
            panel_sha256="a" * 64,
            runtime_sha256="b" * 64,
        )
        evaluate_quality_units_batched(
            panel,
            adapters,
            (unit,),
            evidence_store=first_store,
        )
        changed_store = TeacherBatchEvidenceStore(
            root=Path(directory),
            panel_sha256="a" * 64,
            runtime_sha256="c" * 64,
        )
        evaluate_quality_units_batched(
            panel,
            adapters,
            (unit,),
            evidence_store=changed_store,
        )

        assert all(adapter.calls == 2 for adapter in adapters.values())
        assert changed_store.audit()["hits"] == 0


def test_resume_calls_only_the_provider_that_was_unavailable() -> None:
    panel = load_teacher_panel(PANEL)
    unit = EvaluationUnit(
        unit_id="unit-0",
        text="substantive payload",
        declared_context=None,
        attached_evidence=(),
    )
    adapters = {
        teacher.teacher_id: CountingAdapter(unavailable=index == 2)
        for index, teacher in enumerate(panel.teachers)
    }

    with TemporaryDirectory() as directory:
        store = TeacherBatchEvidenceStore(
            root=Path(directory),
            panel_sha256="a" * 64,
            runtime_sha256="b" * 64,
        )
        evaluate_quality_units_batched(
            panel,
            adapters,
            (unit,),
            evidence_store=store,
        )
        calls_after_partial = {
            teacher_id: adapter.calls for teacher_id, adapter in adapters.items()
        }
        adapters[panel.teachers[2].teacher_id].unavailable = False

        evaluate_quality_units_batched(
            panel,
            adapters,
            (unit,),
            evidence_store=store,
        )

        assert adapters[panel.teachers[0].teacher_id].calls == calls_after_partial[
            panel.teachers[0].teacher_id
        ]
        assert adapters[panel.teachers[1].teacher_id].calls == calls_after_partial[
            panel.teachers[1].teacher_id
        ]
        assert adapters[panel.teachers[2].teacher_id].calls == (
            calls_after_partial[panel.teachers[2].teacher_id] + 1
        )


def test_provider_cache_uses_compact_windows_safe_directory_names() -> None:
    panel = load_teacher_panel(PANEL)
    unit = EvaluationUnit(
        unit_id="unit-with-a-compact-provider-cache-path",
        text="substantive payload",
        declared_context=None,
        attached_evidence=(),
    )
    adapters = {teacher.teacher_id: CountingAdapter() for teacher in panel.teachers}

    with TemporaryDirectory() as directory:
        store = TeacherBatchEvidenceStore(
            root=Path(directory),
            panel_sha256="a" * 64,
            runtime_sha256="b" * 64,
        )
        evaluate_quality_units_batched(panel, adapters, (unit,), evidence_store=store)

        teacher_directories = {path.parent.parent.name for path in Path(directory).rglob("*.json")}
        assert len(teacher_directories) == 3
        assert all(len(name) == 12 for name in teacher_directories)


if __name__ == "__main__":
    test_completed_teacher_batches_survive_before_panel_observation()
    test_runtime_identity_change_cannot_reuse_teacher_evidence()
    test_resume_calls_only_the_provider_that_was_unavailable()
    test_provider_cache_uses_compact_windows_safe_directory_names()
    print("[quality-teacher-batch-cache-v1] durable provider evidence: pass")
