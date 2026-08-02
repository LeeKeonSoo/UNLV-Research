from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_quality_gate import build_development_quality_gate
from development_quality_gate_contract import (
    DevelopmentQualityGateReport,
    QualityGateStatus,
    load_quality_gate_registry,
)


REGISTRY = ROOT / "protocols" / "development_quality_gate_registry_v1.json"
REPORT = ROOT / "validation" / "frozen_contracts" / "development_quality_gate_report_v1.json"


def test_current_e3_report_is_hash_linked_and_fails_closed_on_missing_empirical_effects() -> None:
    registry = load_quality_gate_registry(REGISTRY)
    frozen = DevelopmentQualityGateReport.model_validate_json(REPORT.read_text(encoding="utf-8"))
    replay = build_development_quality_gate(registry)

    assert frozen == replay
    assert frozen.registry_sha256 == registry.identity_sha256()
    assert frozen.status is QualityGateStatus.BLOCKED
    assert frozen.matrix_complete is True
    assert frozen.contract_fixture_excluded is True
    assert frozen.empirical_effect_calibration_complete is False
    assert frozen.common_baseline_empirically_verified is False
    assert frozen.runtime_activation is False
    assert frozen.selector_membership_mutated is False
    assert frozen.benchmark_outcomes_read is False
    assert frozen.utility_read is False
    assert set(frozen.blocker_codes) == {
        "quality_empirical_effect_calibration_missing",
        "quality_empirical_common_baseline_missing",
        "quality_provider_not_active",
        "quality_route_not_evidence_ready:code",
        "quality_route_not_evidence_ready:general",
        "quality_route_not_evidence_ready:math",
    }


def test_route_evidence_records_observed_failures_without_promoting_old_provider_scores() -> None:
    report = DevelopmentQualityGateReport.model_validate_json(REPORT.read_text(encoding="utf-8"))
    by_domain = {item.domain.value: item for item in report.routes}

    assert set(by_domain) == {"code", "math", "general"}
    assert by_domain["code"].route == "code_artifact"
    assert by_domain["math"].route == "mathematical_content"
    assert by_domain["general"].route == "general_prose"
    assert all(item.route_gate_decision == "blocked_source_transfer" for item in by_domain.values())
    assert all(item.empirical_effect_bin_count == 0 for item in by_domain.values())
    assert all(item.effect_calibration_artifact is None for item in by_domain.values())


def test_registry_forbids_fixture_benchmark_utility_and_membership_authority() -> None:
    registry = load_quality_gate_registry(REGISTRY)
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry.contract_fixture_may_satisfy_empirical_gate is False
    assert registry.benchmark_outcomes_available is False
    assert registry.utility_available is False
    assert registry.selector_membership_mutation_allowed is False
    assert payload["required_domains"] == ["code", "math", "general"]
    assert payload["minimum_ordered_bins_per_route"] == 3


if __name__ == "__main__":
    test_current_e3_report_is_hash_linked_and_fails_closed_on_missing_empirical_effects()
    test_route_evidence_records_observed_failures_without_promoting_old_provider_scores()
    test_registry_forbids_fixture_benchmark_utility_and_membership_authority()
    print("[development-quality-gate-v1] empirical evidence boundary: pass")
