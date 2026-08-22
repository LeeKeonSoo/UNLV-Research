from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quality_model_evidence import MissingQualityFallbackEvidenceError
from quality_operating_points import QualityAction, decide_quality_action
from quality_ranker_policy import (
    DistilledPolicyContract,
    distilled_policy_result,
)


def test_distilled_evidence_uses_frozen_fail_threshold_and_ood_requires_fallback() -> None:
    contract = DistilledPolicyContract(
        policy_id="q3_substantive_payload",
        class_labels=("pass", "fail", "abstain"),
        failure_threshold=0.80,
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

    assert decide_quality_action((hard_only,), False).action is QualityAction.NOT_SELECT
    try:
        decide_quality_action((ood,), False)
    except MissingQualityFallbackEvidenceError:
        pass
    else:
        raise AssertionError("OOD evidence must be resolved by the Luna fallback")
    assert ood.decision_source == "distilled_ranker"
    assert ood.reason_codes == ("quality_ranker_ood_abstain",)


if __name__ == "__main__":
    test_distilled_evidence_uses_frozen_fail_threshold_and_ood_requires_fallback()
    print("[quality-ranker-policy-v2] qualified fail and OOD fallback: pass")
