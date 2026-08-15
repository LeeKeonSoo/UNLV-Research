from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_operating_points import CurationMode, QualityAction, decide_quality_action
from quality_ranker_policy import (
    DistilledPolicyContract,
    calibrate_failure_threshold,
    distilled_policy_result,
)


def test_failure_threshold_tightens_when_false_positive_tolerance_decreases() -> None:
    labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    fail_probabilities = np.asarray(
        [0.01, 0.05, 0.15, 0.35, 0.55, 0.45, 0.60, 0.75, 0.90, 0.98],
        dtype=np.float64,
    )

    normal = calibrate_failure_threshold(
        labels,
        fail_probabilities,
        maximum_false_positive_rate=0.0,
        minimum_fail_predictions=2,
    )
    hard = calibrate_failure_threshold(
        labels,
        fail_probabilities,
        maximum_false_positive_rate=0.25,
        minimum_fail_predictions=2,
    )

    assert normal is not None
    assert hard is not None
    assert normal >= hard


def test_distilled_evidence_obeys_normal_hard_and_ood_fail_closed() -> None:
    contract = DistilledPolicyContract(
        policy_id="q3_substantive_payload",
        class_labels=("pass", "fail", "abstain"),
        normal_fail_threshold=0.95,
        hard_fail_threshold=0.80,
        minimum_decision_confidence=0.70,
        ranker_artifact_sha256="a" * 64,
    )

    hard_only = distilled_policy_result(
        contract,
        class_probabilities=np.asarray([0.08, 0.87, 0.05], dtype=np.float64),
        out_of_distribution=False,
    )
    ood = distilled_policy_result(
        contract,
        class_probabilities=np.asarray([0.01, 0.98, 0.01], dtype=np.float64),
        out_of_distribution=True,
    )

    assert decide_quality_action((hard_only,), CurationMode.NORMAL, False).action is QualityAction.NOT_SELECT
    assert decide_quality_action((hard_only,), CurationMode.HARD, False).action is QualityAction.NOT_SELECT
    assert decide_quality_action((ood,), CurationMode.HARD, False).action is QualityAction.NOT_SELECT
    assert ood.decision_source == "distilled_ranker"
    assert ood.reason_codes == ("quality_ranker_ood_abstain",)


if __name__ == "__main__":
    test_failure_threshold_tightens_when_false_positive_tolerance_decreases()
    test_distilled_evidence_obeys_normal_hard_and_ood_fail_closed()
    print("[quality-ranker-policy-v1] calibrated operating points and OOD fail-closed: pass")
