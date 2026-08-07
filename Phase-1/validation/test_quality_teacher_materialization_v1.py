from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_materialization import (
    OBSERVATION_SCHEMA,
    QualityBatchUnavailableError,
    _evaluate_reliably,
    _load_cache,
    materialize_modes,
)
from quality_teacher_unit_runtime import InsufficientTeacherAvailability
from quality_teacher_panel import PanelDecision, PolicyDecision, TeacherVote
from quality_teacher_runtime import PanelPolicyResult


POLICY_IDS = (
    "q1_correctness_evidence",
    "q2_semantic_coherence",
    "q3_substantive_payload",
    "q4_learnable_relations",
)


def _result(policy_id: str, decisions: tuple[PolicyDecision, ...]) -> PanelPolicyResult:
    votes = tuple(
        TeacherVote(
            teacher_id=f"teacher-{index}",
            policy_id=policy_id,
            decision=decision,
            reason_codes=("controlled_reason",),
        )
        for index, decision in enumerate(decisions)
    )
    panel_decision = (
        PanelDecision.FAIL
        if decisions.count(PolicyDecision.FAIL) >= 2
        else PanelDecision.PASS
    )
    second_pass = votes if decisions.count(PolicyDecision.FAIL) == 2 else None
    return PanelPolicyResult(policy_id, panel_decision, votes, second_pass)


def _all_pass() -> tuple[PanelPolicyResult, ...]:
    return tuple(
        _result(policy_id, (PolicyDecision.PASS,) * 3) for policy_id in POLICY_IDS
    )


def test_normal_and_hard_share_evidence_but_use_different_fail_strength() -> None:
    rows = [
        {"chunk_uid": "keep", "text": "substantive payload", "token_proxy": 2},
        {"chunk_uid": "normal-remove", "text": "boilerplate", "token_proxy": 1},
        {"chunk_uid": "hard-only-remove", "text": "fragment", "token_proxy": 1},
    ]
    results = {
        "keep": _all_pass(),
        "normal-remove": (
            _result("q1_correctness_evidence", (PolicyDecision.PASS,) * 3),
            _result("q2_semantic_coherence", (PolicyDecision.PASS,) * 3),
            _result("q3_substantive_payload", (PolicyDecision.FAIL,) * 3),
            _result("q4_learnable_relations", (PolicyDecision.PASS,) * 3),
        ),
        "hard-only-remove": (
            _result("q1_correctness_evidence", (PolicyDecision.PASS,) * 3),
            _result("q2_semantic_coherence", (PolicyDecision.PASS,) * 3),
            _result(
                "q3_substantive_payload",
                (PolicyDecision.FAIL, PolicyDecision.FAIL, PolicyDecision.PASS),
            ),
            _result("q4_learnable_relations", (PolicyDecision.PASS,) * 3),
        ),
    }
    with TemporaryDirectory() as directory:
        report = materialize_modes(
            rows,
            results,
            output_dir=Path(directory),
            scoring_audit={"fixture": True},
        )
        normal = {
            json.loads(line)["chunk_uid"]
            for line in Path(report["modes"]["normal"]["retained_path"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        hard = {
            json.loads(line)["chunk_uid"]
            for line in Path(report["modes"]["hard"]["retained_path"])
            .read_text(encoding="utf-8")
            .splitlines()
        }
        assert normal == {"keep", "hard-only-remove"}
        assert hard == {"keep"}
        assert hard <= normal
        assert report["abstain_action"] == "retain"
        assert report["benchmark_outcomes_read"] is False


def test_unavailable_provider_observation_is_never_reused() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cache.jsonl"
        panel_sha256 = "a" * 64
        runtime_sha256 = "c" * 64
        payload = {
            "schema_version": OBSERVATION_SCHEMA,
            "task_id": "unavailable",
            "teacher_panel_sha256": panel_sha256,
            "quality_runtime_sha256": runtime_sha256,
            "chunk_uid": "chunk",
            "text_sha256": "b" * 64,
            "available_teacher_ids": ["teacher-0"],
            "unavailable_teacher_ids": ["teacher-1", "teacher-2"],
            "policy_results": [],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        cache, ignored = _load_cache(path, panel_sha256, runtime_sha256)
        assert cache == {}
        assert ignored == 1


def test_observation_cache_rejects_runtime_identity_mismatch() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "cache.jsonl"
        payload = {
            "schema_version": OBSERVATION_SCHEMA,
            "task_id": "cached",
            "teacher_panel_sha256": "a" * 64,
            "quality_runtime_sha256": "b" * 64,
            "chunk_uid": "chunk",
            "text_sha256": "c" * 64,
            "available_teacher_ids": ["teacher-0", "teacher-1"],
            "unavailable_teacher_ids": ["teacher-2"],
            "policy_results": [{}, {}, {}, {}],
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

        try:
            _load_cache(path, "a" * 64, "d" * 64)
        except RuntimeError as error:
            assert "runtime identity mismatch" in str(error)
        else:
            raise AssertionError("changed Quality runtime must invalidate observations")


def test_rate_limit_retry_uses_cooldown_scale_backoff() -> None:
    attempts = 0
    sleeps: list[float] = []
    expected = ("recovered",)

    def evaluator(panel, adapters, units):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise InsufficientTeacherAvailability(
                unit_id="fixture",
                available_teachers=1,
            )
        return expected

    observed = _evaluate_reliably(
        panel=None,
        adapters={},
        units=(),
        retry_delays_seconds=(30.0, 60.0, 120.0),
        evaluator=evaluator,
        sleep_fn=sleeps.append,
    )

    assert observed == expected
    assert attempts == 3
    assert sleeps == [30.0, 60.0]


def test_exhausted_provider_retry_raises_only_the_typed_resume_error() -> None:
    def unavailable(panel, adapters, units):
        raise InsufficientTeacherAvailability(
            unit_id="fixture",
            available_teachers=1,
        )

    try:
        _evaluate_reliably(
            panel=None,
            adapters={},
            units=(),
            retry_delays_seconds=(1.0,),
            evaluator=unavailable,
            sleep_fn=lambda _: None,
        )
    except QualityBatchUnavailableError as error:
        assert error.attempts == 2
    else:
        raise AssertionError("Expected the typed resumable provider error")


def test_reliable_evaluation_forwards_provider_evidence_store() -> None:
    expected_store = object()

    def evaluator(panel, adapters, units, *, evidence_store):
        assert evidence_store is expected_store
        return ("cached",)

    observed = _evaluate_reliably(
        panel=None,
        adapters={},
        units=(),
        evidence_store=expected_store,
        evaluator=evaluator,
    )

    assert observed == ("cached",)


if __name__ == "__main__":
    test_normal_and_hard_share_evidence_but_use_different_fail_strength()
    test_unavailable_provider_observation_is_never_reused()
    test_observation_cache_rejects_runtime_identity_mismatch()
    test_rate_limit_retry_uses_cooldown_scale_backoff()
    test_exhausted_provider_retry_raises_only_the_typed_resume_error()
    test_reliable_evaluation_forwards_provider_evidence_store()
    print("[quality-teacher-materialization-v1] shared evidence and deletion authority: pass")
