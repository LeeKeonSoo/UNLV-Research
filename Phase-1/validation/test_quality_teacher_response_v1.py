#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_teacher_panel import PolicyDecision
from quality_teacher_response import TeacherResponseAttempt, resolve_teacher_response


def test_valid_enum_response_is_parsed_without_retry() -> None:
    # Given: the first response follows the machine-consumed schema.
    attempt = TeacherResponseAttempt(
        teacher_id="local",
        policy_id="q3_substantive_payload",
        first_raw='{"decision":"fail","reason_codes":["pure_navigation"]}',
        retry_raw=None,
    )

    # When: the response boundary resolves the attempt.
    vote = resolve_teacher_response(attempt)

    # Then: the typed decision and reason code are preserved.
    assert vote.decision is PolicyDecision.FAIL
    assert vote.reason_codes == ("pure_navigation",)


def test_invalid_boolean_decision_uses_valid_retry() -> None:
    # Given: the observed local-model failure is followed by a valid retry.
    attempt = TeacherResponseAttempt(
        teacher_id="local",
        policy_id="q3_substantive_payload",
        first_raw='{"decision":false,"reason_codes":["pure_navigation"]}',
        retry_raw='{"decision":"fail","reason_codes":["pure_navigation"]}',
    )

    # When: the response boundary resolves the attempt.
    vote = resolve_teacher_response(attempt)

    # Then: only the schema-valid retry becomes evidence.
    assert vote.decision is PolicyDecision.FAIL
    assert vote.reason_codes == ("pure_navigation",)


def test_two_invalid_responses_fail_closed_to_abstain() -> None:
    # Given: both the first response and the single retry violate the schema.
    attempt = TeacherResponseAttempt(
        teacher_id="local",
        policy_id="q3_substantive_payload",
        first_raw="not-json",
        retry_raw='{"decision":false,"reason_codes":[]}',
    )

    # When: the response boundary resolves the attempt.
    vote = resolve_teacher_response(attempt)

    # Then: malformed output cannot become a pass or fail label.
    assert vote.decision is PolicyDecision.ABSTAIN
    assert vote.reason_codes == ("invalid_teacher_response_schema",)


def test_sentence_reason_code_requires_schema_retry() -> None:
    # Given: the first output uses explanatory prose where a reason code is required.
    attempt = TeacherResponseAttempt(
        teacher_id="local",
        policy_id="q3_substantive_payload",
        first_raw=(
            '{"decision":"fail","reason_codes":'
            '["The unit has no substantive content beyond boilerplate."]}'
        ),
        retry_raw='{"decision":"fail","reason_codes":["no_substantive_payload"]}',
    )

    # When: the response boundary resolves the attempt.
    vote = resolve_teacher_response(attempt)

    # Then: prose cannot silently become a machine reason code.
    assert vote.decision is PolicyDecision.FAIL
    assert vote.reason_codes == ("no_substantive_payload",)


if __name__ == "__main__":
    test_valid_enum_response_is_parsed_without_retry()
    test_invalid_boolean_decision_uses_valid_retry()
    test_two_invalid_responses_fail_closed_to_abstain()
    test_sentence_reason_code_requires_schema_retry()
    print("[quality-teacher-response-v1] schema retry and abstention: pass")
