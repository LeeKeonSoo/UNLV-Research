from __future__ import annotations

import hashlib
import json
from pathlib import Path

from development_corpus_inventory_contract import DevelopmentCorpusInventoryManifest, InventoryStatus
from development_quality_gate_contract import (
    DevelopmentQualityGateError,
    DevelopmentQualityGateRegistry,
    DevelopmentQualityGateReport,
    EmpiricalQualityEffectBundle,
    QualityGateStatus,
    QualityRouteEvidence,
    hash_json,
)
from model_provider_contract import ProviderLifecycle, ProviderRole, load_provider_registry
from quality_effect_calibration import EffectCalibrationBundle, EffectInterval, EvidenceBin, calibrate_effect_bins


ROOT = Path(__file__).resolve().parent


def _path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frozen(path: Path, expected_sha256: str, reason: str) -> bytes:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise DevelopmentQualityGateError(reason)
    return path.read_bytes()


def _calibrate(bundle: EmpiricalQualityEffectBundle):
    return calibrate_effect_bins(
        EffectCalibrationBundle(
            provider_id=bundle.provider_id,
            provider_identity_sha256=bundle.provider_identity_sha256,
            effect_metric_id=bundle.effect_metric_id,
            effect_metric_artifact_sha256=bundle.effect_metric_artifact_sha256,
            common_baseline_artifact_sha256=bundle.common_baseline_artifact_sha256,
            bins=tuple(
                EvidenceBin(
                    route=bundle.route,
                    bin_id=item.bin_id,
                    bin_order=item.bin_order,
                    development=EffectInterval(**item.development.model_dump()),
                    heldout=EffectInterval(**item.heldout.model_dump()),
                    artifact_sha256=item.artifact_sha256,
                )
                for item in bundle.bins
            ),
            provider_training_source_groups=frozenset(bundle.provider_training_source_groups),
            development_source_groups=frozenset(bundle.development_source_groups),
            heldout_source_groups=frozenset(bundle.heldout_source_groups),
            all_arms_share_common_baseline=bundle.all_arms_share_common_baseline,
            common_baseline_disjoint_from_all_bins=bundle.common_baseline_disjoint_from_all_bins,
            external_results_hidden=bundle.external_results_hidden,
            provider_bias_stress_passed=bundle.provider_bias_stress_passed,
            route_holdout_stress_passed=bundle.route_holdout_stress_passed,
        )
    )


def build_development_quality_gate(registry: DevelopmentQualityGateRegistry) -> DevelopmentQualityGateReport:
    manifest_path = _path(registry.inventory_manifest_path)
    manifest = DevelopmentCorpusInventoryManifest.model_validate_json(
        _read_frozen(
            manifest_path,
            registry.inventory_manifest_file_sha256,
            "quality_inventory_manifest_file_mismatch",
        )
    )
    if (
        manifest.status is not InventoryStatus.ADMITTED
        or manifest.manifest_sha256 != registry.inventory_manifest_sha256
    ):
        raise DevelopmentQualityGateError("quality_inventory_manifest_not_admitted")

    route_path = _path(registry.route_evidence_gate_path)
    route_gate = json.loads(
        _read_frozen(route_path, registry.route_evidence_gate_file_sha256, "quality_route_gate_file_mismatch")
    )
    provider_path = _path(registry.provider_registry_path)
    _read_frozen(
        provider_path,
        registry.provider_registry_file_sha256,
        "quality_provider_registry_file_mismatch",
    )
    providers = load_provider_registry(provider_path)
    fixture_path = _path(registry.contract_fixture_path)
    fixture = json.loads(
        _read_frozen(fixture_path, registry.contract_fixture_file_sha256, "quality_contract_fixture_file_mismatch")
    )
    contract_fixture_excluded = (
        bool(fixture.get("contract_fixture_only_not_empirical_evidence"))
        and not registry.contract_fixture_may_satisfy_empirical_gate
    )

    quality_providers = tuple(provider for provider in providers.providers if provider.role is ProviderRole.QUALITY)
    active_providers = tuple(
        provider
        for provider in quality_providers
        if provider.lifecycle is ProviderLifecycle.ACTIVE and provider.policy_contribution_authority
    )
    provider_active = bool(active_providers)
    provider_by_id = {provider.provider_id: provider for provider in quality_providers}
    route_gate_by_route = {item["route"]: item for item in route_gate.get("routes", ())}
    bundle_by_domain = {item.domain: item for item in registry.empirical_effect_bundles}
    route_evidence: list[QualityRouteEvidence] = []

    for requirement in registry.required_routes:
        blockers: list[str] = []
        gate = route_gate_by_route.get(requirement.route)
        gate_decision = "missing" if gate is None else str(gate.get("decision", "missing"))
        route_ready = gate_decision == "evidence_ready_candidate"
        if not route_ready:
            blockers.append(f"quality_route_not_evidence_ready:{requirement.domain.value}")
        reference = bundle_by_domain.get(requirement.domain)
        calibration_passed = False
        provider_id = None
        provider_lifecycle = None
        provider_authority = False
        effect_count = 0
        if reference is not None:
            artifact_path = _path(reference.artifact_path)
            bundle = EmpiricalQualityEffectBundle.model_validate_json(
                _read_frozen(
                    artifact_path,
                    reference.artifact_file_sha256,
                    f"quality_effect_bundle_hash_mismatch:{requirement.domain.value}",
                )
            )
            if bundle.domain != requirement.domain or bundle.route != requirement.route:
                raise DevelopmentQualityGateError(f"quality_effect_bundle_scope_mismatch:{requirement.domain.value}")
            provider_id = bundle.provider_id
            provider = provider_by_id.get(bundle.provider_id)
            provider_valid = False
            if provider is not None:
                provider_lifecycle = provider.lifecycle.value
                provider_authority = provider.policy_contribution_authority
                provider_valid = (
                    provider.lifecycle is ProviderLifecycle.ACTIVE
                    and provider.policy_contribution_authority
                    and provider.identity_sha256() == bundle.provider_identity_sha256
                    and requirement.route in provider.supported_routes
                )
            effect_count = len(bundle.bins)
            calibration_passed = (
                provider_valid
                and _calibrate(bundle).passed
                and effect_count >= registry.minimum_ordered_bins_per_route
            )
            if not provider_valid:
                blockers.append(f"quality_provider_not_active:{requirement.domain.value}")
            if not calibration_passed:
                blockers.append(f"quality_effect_calibration_failed:{requirement.domain.value}")
        route_evidence.append(
            QualityRouteEvidence(
                domain=requirement.domain,
                route=requirement.route,
                route_gate_decision=gate_decision,
                route_evidence_ready=route_ready,
                provider_id=provider_id,
                provider_lifecycle=provider_lifecycle,
                provider_policy_contribution_authority=provider_authority,
                effect_calibration_artifact=None if reference is None else reference.artifact_path,
                empirical_effect_bin_count=effect_count,
                calibration_passed=calibration_passed,
                blocker_codes=tuple(sorted(blockers)),
            )
        )

    matrix_complete = (
        len(route_evidence) == len(registry.required_domains)
        and {item.domain for item in route_evidence} == set(registry.required_domains)
    )
    calibration_complete = bool(route_evidence) and all(item.calibration_passed for item in route_evidence)
    common_baseline_verified = False
    if calibration_complete and registry.common_baseline_artifact_sha256 is not None:
        bundles = tuple(
            EmpiricalQualityEffectBundle.model_validate_json(_path(item.artifact_path).read_text(encoding="utf-8"))
            for item in registry.empirical_effect_bundles
        )
        common_baseline_verified = all(
            item.common_baseline_artifact_sha256 == registry.common_baseline_artifact_sha256
            and item.all_arms_share_common_baseline
            and item.common_baseline_disjoint_from_all_bins
            for item in bundles
        )

    blockers = [code for item in route_evidence for code in item.blocker_codes]
    if not provider_active:
        blockers.append("quality_provider_not_active")
    if not calibration_complete:
        blockers.append("quality_empirical_effect_calibration_missing")
    if not common_baseline_verified:
        blockers.append("quality_empirical_common_baseline_missing")
    if not contract_fixture_excluded:
        blockers.append("quality_contract_fixture_not_excluded")
    if not matrix_complete:
        blockers.append("quality_route_matrix_incomplete")
    payload = {
        "schema_version": "development-quality-gate-report-v1",
        "registry_sha256": registry.identity_sha256(),
        "inventory_manifest_sha256": manifest.manifest_sha256,
        "inventory_manifest_file_sha256": registry.inventory_manifest_file_sha256,
        "routes": [item.model_dump(mode="json") for item in route_evidence],
        "matrix_complete": matrix_complete,
        "contract_fixture_excluded": contract_fixture_excluded,
        "provider_active": provider_active,
        "empirical_effect_calibration_complete": calibration_complete,
        "common_baseline_empirically_verified": common_baseline_verified,
        "blocker_codes": sorted(set(blockers)),
        "runtime_activation": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "selector_membership_mutated": False,
    }
    return DevelopmentQualityGateReport(
        status=QualityGateStatus.PASSED if not blockers else QualityGateStatus.BLOCKED,
        report_sha256=hash_json(payload),
        **payload,
    )


__all__ = ["build_development_quality_gate"]
