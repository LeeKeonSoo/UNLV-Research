#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework_objects import FrameworkObjectError, ObjectRegistry, load_framework_objects

MANIFEST = ROOT / "configs" / "curation_framework_v1.json"
REGISTRY = ROOT / "configs" / "framework_objects_v1.json"


def test_registry_parses_complete_core_metric_policy_method_lineage() -> None:
    # Given: the central manifest and object registry.
    registry = load_framework_objects(MANIFEST, REGISTRY)

    # When: every policy lineage is projected.
    metrics = {metric.id: metric for metric in registry.metrics}
    methods = {method.id: method for method in registry.methods}

    # Then: every policy is owned by one Core and resolves all dependencies.
    assert {policy.core_id.value for policy in registry.policies} == {
        "validity",
        "redundancy",
        "quality",
        "coverage",
    }
    for policy in registry.policies:
        for metric_id in policy.metric_ids:
            metric = metrics[metric_id]
            assert metric.core_id is policy.core_id
            assert metric.method_id in methods
    teacher_panel = next(
        policy for policy in registry.policies if policy.id == "quality.teacher_panel_v2"
    )
    assert tuple(item.path for item in teacher_panel.evidence) == (
        "configs/quality_teacher_panel_v2.json",
    )
    assert teacher_panel.lifecycle.value == "blocked"
    assert teacher_panel.decision_authority.value == "quality_decision"
    coverage = next(
        policy for policy in registry.policies if policy.id == "coverage.representative_guard"
    )
    assert "coverage.semantic_support_extinction" in coverage.metric_ids
    assert tuple(item.path for item in coverage.evidence) == (
        "configs/semantic_coverage_v3.json",
    )


def test_provider_output_cannot_have_direct_deletion_authority() -> None:
    # Given: a valid registry payload.
    registry = load_framework_objects(MANIFEST, REGISTRY)
    payload = registry.model_dump(mode="json")
    payload["providers"][0]["direct_deletion_authority"] = True

    # When / Then: the typed boundary rejects provider-owned deletion.
    try:
        ObjectRegistry.model_validate(payload)
    except ValidationError as error:
        assert "direct_deletion_authority" in str(error)
    else:
        raise AssertionError("Provider output must not delete without a Policy")


def test_cross_core_metric_reference_is_rejected() -> None:
    # Given: a Quality policy rewritten to consume a Redundancy metric.
    registry = load_framework_objects(MANIFEST, REGISTRY)
    payload = registry.model_dump(mode="json")
    quality_policy = next(item for item in payload["policies"] if item["core_id"] == "quality")
    quality_policy["metric_ids"] = ["redundancy.exact_text_identity"]

    # When / Then: the registry cannot represent cross-Core authority leakage.
    try:
        ObjectRegistry.model_validate(payload)
    except ValidationError as error:
        assert "framework_policy_metric_core_mismatch" in str(error)
    else:
        raise AssertionError("A Policy must consume Metrics from its own Core")


def test_registry_is_bound_to_exact_framework_manifest_bytes() -> None:
    # Given: a registry whose root hash is replaced.
    registry = load_framework_objects(MANIFEST, REGISTRY)
    payload = registry.model_dump(mode="json")
    payload["framework_manifest_sha256"] = "f" * 64
    changed = ObjectRegistry.model_validate(payload)

    # When / Then: the boundary loader rejects stale or foreign roots.
    try:
        load_framework_objects(MANIFEST, changed)
    except FrameworkObjectError as error:
        assert "framework_manifest_identity_mismatch" in str(error)
    else:
        raise AssertionError("Registry identity must be rooted in the central manifest")


if __name__ == "__main__":
    test_registry_parses_complete_core_metric_policy_method_lineage()
    test_provider_output_cannot_have_direct_deletion_authority()
    test_cross_core_metric_reference_is_rejected()
    test_registry_is_bound_to_exact_framework_manifest_bytes()
    print("[framework-objects-v1] typed lineage and authority boundary: pass")
