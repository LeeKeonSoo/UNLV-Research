#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_objects import CoreId, StageId
from stage_permissions import (
    StageInputRequest,
    StagePermissionError,
    authorize_stage_input,
    load_stage_authority,
)

MANIFEST = ROOT / "configs" / "curation_framework_v1.json"


def test_each_stage_accepts_only_its_declared_core_and_inputs() -> None:
    # Given: the frozen Stage authority registry.
    authority = load_stage_authority(MANIFEST)
    request = StageInputRequest(
        stage_id=StageId.STAGE_B,
        core_id=CoreId.REDUNDANCY,
        supplied_categories=("stage_a_survivors", "stable_identifiers"),
    )

    # When: a Stage-B Redundancy request uses declared inputs.
    authorized = authorize_stage_input(authority, request)

    # Then: authorization is explicit and immutable.
    assert authorized.stage_id is StageId.STAGE_B
    assert authorized.core_id is CoreId.REDUNDANCY
    assert authorized.supplied_categories == request.supplied_categories


def test_runtime_forbidden_input_is_rejected_at_every_stage() -> None:
    # Given: the three public stages and their owning Cores.
    authority = load_stage_authority(MANIFEST)
    owners = {
        StageId.STAGE_A: CoreId.VALIDITY,
        StageId.STAGE_B: CoreId.QUALITY,
        StageId.STAGE_C: CoreId.COVERAGE,
    }

    # When / Then: Utility cannot cross any Stage boundary.
    for stage_id, core_id in owners.items():
        request = StageInputRequest(
            stage_id=stage_id,
            core_id=core_id,
            supplied_categories=("utility",),
        )
        try:
            authorize_stage_input(authority, request)
        except StagePermissionError as error:
            assert error.reason_code == "stage_runtime_forbidden_input"
        else:
            raise AssertionError(f"Utility entered {stage_id.value}")


def test_stage_c_rejects_new_quality_or_ranking_inputs() -> None:
    # Given: a Coverage materialization request.
    authority = load_stage_authority(MANIFEST)

    # When / Then: Stage C cannot become a hidden selector.
    for category in ("new_quality_score", "new_ranking_objective", "quota_based_restoration"):
        request = StageInputRequest(
            stage_id=StageId.STAGE_C,
            core_id=CoreId.COVERAGE,
            supplied_categories=(category,),
        )
        try:
            authorize_stage_input(authority, request)
        except StagePermissionError as error:
            assert error.reason_code == "stage_local_forbidden_input"
        else:
            raise AssertionError(f"Stage C accepted {category}")


def test_cross_stage_core_request_is_rejected() -> None:
    # Given: Quality is incorrectly routed to Stage A.
    authority = load_stage_authority(MANIFEST)
    request = StageInputRequest(
        stage_id=StageId.STAGE_A,
        core_id=CoreId.QUALITY,
        supplied_categories=("raw_text",),
    )

    # When / Then: Stage ownership cannot be overridden by the caller.
    try:
        authorize_stage_input(authority, request)
    except StagePermissionError as error:
        assert error.reason_code == "stage_core_authority_mismatch"
    else:
        raise AssertionError("A Core entered a Stage it does not own")


def test_undeclared_input_fails_closed() -> None:
    # Given: metadata that is neither allowed nor globally named.
    authority = load_stage_authority(MANIFEST)
    request = StageInputRequest(
        stage_id=StageId.STAGE_B,
        core_id=CoreId.QUALITY,
        supplied_categories=("opaque_user_metadata",),
    )

    # When / Then: undeclared input is not silently ignored.
    try:
        authorize_stage_input(authority, request)
    except StagePermissionError as error:
        assert error.reason_code == "stage_undeclared_input"
    else:
        raise AssertionError("Unknown metadata crossed the Stage boundary")


if __name__ == "__main__":
    test_each_stage_accepts_only_its_declared_core_and_inputs()
    test_runtime_forbidden_input_is_rejected_at_every_stage()
    test_stage_c_rejects_new_quality_or_ranking_inputs()
    test_cross_stage_core_request_is_rejected()
    test_undeclared_input_fails_closed()
    print("[stage-permissions-v1] A/B/C input and Core authority: pass")
