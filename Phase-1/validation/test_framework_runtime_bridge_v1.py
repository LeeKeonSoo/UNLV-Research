#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_objects import CoreId, StageId
from framework_runtime_bridge import (
    RuntimeStageRequest,
    authorize_runtime_stage,
    load_runtime_foundation,
)


def test_runtime_foundation_loads_every_redesign_contract() -> None:
    # Given: the project root used by the production entry point.
    foundation = load_runtime_foundation(ROOT)

    # When / Then: all identities are verified before input data is read.
    assert foundation.schema_version == "framework-runtime-foundation-v1"
    assert foundation.bridge.new_v1_policy_activation is True
    assert foundation.bridge.curated_output_equivalence_required is False
    assert {profile.id.value for profile in foundation.profiles.profiles} == {"normal", "hard"}
    assert all(not profile.release_enabled for profile in foundation.profiles.profiles)


def test_final_runtime_has_no_declared_policy_left_blocked_or_legacy_mapped() -> None:
    # Given: the compatibility bridge and typed Policy registry.
    foundation = load_runtime_foundation(ROOT)
    lifecycle_by_id = {policy.id: policy.lifecycle.value for policy in foundation.objects.policies}

    # When / Then: the final experimental runtime has no hidden blocked branch.
    assert foundation.bridge.blocked_v1_policy_ids == ()
    assert foundation.bridge.legacy_policy_mappings == ()
    assert lifecycle_by_id["redundancy.symmetric_near_duplicate_candidate"] == "candidate"
    assert lifecycle_by_id["quality.distilled_ranker_v1"] == "candidate"


def test_runtime_stage_ticket_is_issued_from_central_permissions() -> None:
    # Given: a Stage-B Redundancy invocation from the legacy-compatible kernel.
    foundation = load_runtime_foundation(ROOT)
    request = RuntimeStageRequest(
        stage_id=StageId.STAGE_B,
        core_id=CoreId.REDUNDANCY,
        supplied_categories=(
            "stage_a_survivors",
            "deterministic_normalized_text",
            "stable_identifiers",
            "runtime_local_structural_evidence",
        ),
    )

    # When: the bridge asks the central authority for a ticket.
    ticket = authorize_runtime_stage(foundation, request)

    # Then: the ticket is auditable and carries no selector output.
    assert ticket.stage_id is StageId.STAGE_B
    assert ticket.core_id is CoreId.REDUNDANCY
    assert ticket.authorization == "central_stage_permission_granted"
    assert ticket.selector_decision is None


if __name__ == "__main__":
    test_runtime_foundation_loads_every_redesign_contract()
    test_final_runtime_has_no_declared_policy_left_blocked_or_legacy_mapped()
    test_runtime_stage_ticket_is_issued_from_central_permissions()
    print("[framework-runtime-bridge-v1] identities, active candidates, Stage tickets: pass")
