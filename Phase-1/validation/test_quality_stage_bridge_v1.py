#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_operating_points import CurationMode
from quality_stage_bridge import apply_coverage_veto, propose_quality_selections
from quality_teacher_panel import PanelDecision, PolicyDecision, TeacherVote
from quality_teacher_runtime import PanelPolicyResult


def _failed_result() -> PanelPolicyResult:
    votes = tuple(
        TeacherVote(
            teacher_id=f"teacher-{index}",
            policy_id="q3_substantive_payload",
            decision=PolicyDecision.FAIL,
            reason_codes=("boilerplate_only",),
        )
        for index in range(3)
    )
    return PanelPolicyResult("q3_substantive_payload", PanelDecision.FAIL, votes, None)


def test_stage_b_selects_only_quality_passes_and_stage_c_may_restore() -> None:
    proposals = propose_quality_selections(
        {"keep-me": (_failed_result(),), "remove-me": (_failed_result(),)},
        CurationMode.NORMAL,
    )

    final = apply_coverage_veto(proposals, protected_uids={"keep-me"})

    assert final["remove-me"].final_action == "not_select"
    assert final["keep-me"].final_action == "retain"
    assert final["keep-me"].stage_b_reason_code == "quality_normal_qualified_fail"
    assert final["keep-me"].stage_c_reason_code == "coverage_veto_retain"


if __name__ == "__main__":
    test_stage_b_selects_only_quality_passes_and_stage_c_may_restore()
    print("[quality-stage-bridge-v1] positive selection and Stage C Coverage veto: pass")
