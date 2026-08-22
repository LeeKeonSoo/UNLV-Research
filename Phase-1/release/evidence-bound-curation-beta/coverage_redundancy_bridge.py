from __future__ import annotations

from coverage_contract import RepresentativeFamily
from redundancy_mode_policy import RedundancyPlan


def coverage_families_from_redundancy_plan(
    plan: RedundancyPlan,
) -> tuple[RepresentativeFamily, ...]:
    return tuple(
        RepresentativeFamily(
            family_id=family.family_id,
            member_uids=frozenset(family.member_uids),
            evidence_artifact_sha256=family.evidence_sha256,
            preferred_representative_uid=family.representative_uid,
        )
        for family in plan.families
    )
