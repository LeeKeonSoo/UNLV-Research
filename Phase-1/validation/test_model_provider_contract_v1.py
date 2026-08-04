#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_provider_contract import (
    ProviderContractError,
    ProviderLifecycle,
    ProviderManifest,
    load_provider_registry,
    validate_provider_replacement,
)


REGISTRY = ROOT / "configs" / "model_provider_registry_v1.json"


def test_registry_exposes_replaceable_roles_without_direct_deletion_authority() -> None:
    registry = load_provider_registry(REGISTRY)

    assert registry.schema_version == "model-provider-registry-v1"
    assert set(registry.slots) == {
        "quality",
        "semantic",
        "diagnostic_validity",
        "content_router",
    }
    assert all(not provider.direct_deletion_authority for provider in registry.providers)
    assert all(not slot.provider_output_alone_may_delete for slot in registry.slots.values())


def test_teacher_panel_is_runtime_experiment_and_embedding_remains_audit_only() -> None:
    registry = load_provider_registry(REGISTRY)
    providers = {provider.provider_id: provider for provider in registry.providers}

    teacher_panel = providers["quality-teacher-panel-v2"]
    embedding = providers["qwen3-embedding-0.6b-semantic-candidate"]
    assert teacher_panel.lifecycle is ProviderLifecycle.RUNTIME_EXPERIMENT
    assert embedding.lifecycle is ProviderLifecycle.AUDIT_ONLY
    assert teacher_panel.policy_contribution_authority is True
    assert embedding.policy_contribution_authority is False


def test_changed_provider_fingerprint_cannot_inherit_calibration() -> None:
    registry = load_provider_registry(REGISTRY)
    current = next(provider for provider in registry.providers if provider.role.value == "semantic")
    changed_payload = current.model_dump(mode="json")
    changed_payload["artifacts"][0]["revision"] = "different-frozen-revision"
    changed_payload["artifacts"][0]["artifact_sha256"] = "b" * 64
    changed_payload["lifecycle"] = ProviderLifecycle.CALIBRATED.value
    changed_payload["calibration"] = {
        "artifact_path": "outputs/calibration/semantic.json",
        "artifact_sha256": "a" * 64,
        "scope_id": "frozen-development-scope",
    }
    changed = ProviderManifest.model_validate(changed_payload)

    try:
        validate_provider_replacement(current, changed)
    except ProviderContractError as error:
        assert "audit_only" in str(error)
    else:
        raise AssertionError("A replacement provider must not inherit calibration")


if __name__ == "__main__":
    test_registry_exposes_replaceable_roles_without_direct_deletion_authority()
    test_teacher_panel_is_runtime_experiment_and_embedding_remains_audit_only()
    test_changed_provider_fingerprint_cannot_inherit_calibration()
    print("[model-provider-contract-v1] replacement and fail-closed gates: pass")
