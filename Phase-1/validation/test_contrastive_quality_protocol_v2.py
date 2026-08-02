#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contrastive_quality_contract import (
    ContrastiveProtocolError,
    ContrastiveQualityProtocol,
    LossObservation,
    compute_contrastive_gaps,
    load_contrastive_protocol,
    validate_protocol_replacement,
)

PROTOCOL = ROOT / "configs" / "contrastive_quality_protocol_v2.json"


def _ready_protocol() -> ContrastiveQualityProtocol:
    blocked = load_contrastive_protocol(PROTOCOL)
    payload = blocked.model_dump(mode="json")
    payload["lifecycle"] = "development_passed"
    for model in payload["models"]:
        if model["role"] == "quality_reference":
            model["role_qualification"] = "validated_reference_pool"
            model["training_distribution_artifact_sha256"] = "a" * 64
            model["quantization_validation_artifact_sha256"] = "5" * 64
        if model["role"] == "background":
            model["provider_id"] = "broad-background-v1"
            model["model_id"] = "example/background-model"
            model["revision"] = "frozen-background-revision"
            model["artifact_sha256"] = "b" * 64
            model["precision"] = "bfloat16"
            model["role_qualification"] = "broad_background"
    payload["calibration"] = {
        "status": "ready",
        "blocker_codes": [],
        "development_corpus_sha256": "c" * 64,
        "sample_count": 3000,
        "supported_routes": ["code_artifact", "mathematical_content", "general_prose"],
        "common_baseline_sha256": "d" * 64,
        "sensitivity_arm_sha256s": ["e" * 64, "f" * 64],
        "disjointness_artifact_sha256": "1" * 64,
        "calibration_artifact_sha256": "2" * 64,
        "effect_bin_artifact_sha256": "3" * 64,
        "external_evidence_sha256": "4" * 64,
    }
    return ContrastiveQualityProtocol.model_validate(payload)


def test_blocked_protocol_has_three_roles_and_no_runtime_authority() -> None:
    # Given: the first v2 contrastive protocol.
    protocol = load_contrastive_protocol(PROTOCOL)

    # When / Then: all scientific roles exist but incomplete evidence blocks use.
    assert {model.role.value for model in protocol.models} == {
        "target",
        "quality_reference",
        "background",
    }
    assert protocol.calibration.status.value == "blocked"
    assert protocol.runtime_authority is False
    assert protocol.direct_deletion_authority is False


def test_gap_directions_are_not_collapsed_into_a_quality_score() -> None:
    # Given: losses from the three declared roles.
    observation = LossObservation(target_nll=3.0, quality_reference_nll=1.0, background_nll=0.5)

    # When: directional contrastive evidence is computed.
    gaps = compute_contrastive_gaps(observation)

    # Then: learnability is keep evidence and alignment is removal-candidate evidence.
    assert gaps.learnability_gap == 2.0
    assert gaps.alignment_gap == 0.5
    assert gaps.scalar_quality_score is None


def test_generic_larger_base_cannot_qualify_as_quality_reference() -> None:
    # Given: a protocol marked ready without a validated reference distribution.
    protocol = _ready_protocol()
    payload = protocol.model_dump(mode="json")
    reference = next(model for model in payload["models"] if model["role"] == "quality_reference")
    reference["role_qualification"] = "unqualified_generic_base"
    reference["training_distribution_artifact_sha256"] = None

    # When / Then: model size alone cannot create Quality authority.
    try:
        ContrastiveQualityProtocol.model_validate(payload)
    except ValidationError as error:
        assert "contrastive_quality_reference_unqualified" in str(error)
    else:
        raise AssertionError("A generic larger base model was treated as Quality truth")


def test_common_baseline_must_be_disjoint_from_every_sensitivity_arm() -> None:
    # Given: a ready calibration with one arm reusing the baseline artifact.
    protocol = _ready_protocol()
    payload = protocol.model_dump(mode="json")
    baseline = payload["calibration"]["common_baseline_sha256"]
    payload["calibration"]["sensitivity_arm_sha256s"][0] = baseline

    # When / Then: the common-baseline audit fails closed.
    try:
        ContrastiveQualityProtocol.model_validate(payload)
    except ValidationError as error:
        assert "contrastive_common_baseline_not_disjoint" in str(error)
    else:
        raise AssertionError("A sensitivity arm reused the common baseline")


def test_provider_change_invalidates_ready_calibration() -> None:
    # Given: a calibrated protocol and a replacement target revision.
    current = _ready_protocol()
    payload = current.model_dump(mode="json")
    target = next(model for model in payload["models"] if model["role"] == "target")
    target["revision"] = "replacement-revision"
    replacement = ContrastiveQualityProtocol.model_validate(payload)

    # When / Then: calibration cannot be inherited across provider identity.
    try:
        validate_protocol_replacement(current, replacement)
    except ContrastiveProtocolError as error:
        assert error.reason_code == "contrastive_provider_change_requires_recalibration"
    else:
        raise AssertionError("Changed provider identity inherited calibration")


if __name__ == "__main__":
    test_blocked_protocol_has_three_roles_and_no_runtime_authority()
    test_gap_directions_are_not_collapsed_into_a_quality_score()
    test_generic_larger_base_cannot_qualify_as_quality_reference()
    test_common_baseline_must_be_disjoint_from_every_sensitivity_arm()
    test_provider_change_invalidates_ready_calibration()
    print("[contrastive-quality-protocol-v2] roles, directions, calibration invalidation: pass")
