#!/usr/bin/env python3
"""Build fixture-based behavior evidence for each Core axis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from data_eval_common import OUTPUT_DIR, load_json, save_json
from ingestion.code_selection import local_stage_b_features, score_stage_b, select_stage_b
from policy.subsets import _coverage_retention, _distribution_bucket_support
from quality.reference_quality import load_reference_quality_model
from signals.core import CoreMetricScorer


DEFAULT_CONSTRUCT_REVIEW = OUTPUT_DIR / "validation" / "core_construct_validity_review.json"
DEFAULT_SELECTOR_LEAKAGE = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_CONFIRMATORY_DECISION = OUTPUT_DIR / "validation" / "code_domain_v2_confirmatory_decision_report.json"
DEFAULT_STAGE_B_PROXY_FIXTURES = Path("validation") / "fixtures" / "temporal_code_stage_b_proxy_cases.json"
DEFAULT_STAGE0_HAZARD_BENCHMARK = OUTPUT_DIR / "validation" / "stage0_hazard_benchmark_report.json"
DEFAULT_STAGE0_DETECTOR_VALIDATION = OUTPUT_DIR / "validation" / "stage0_detector_validation_report.json"
DEFAULT_STAGE0_DETECTOR_HELDOUT = OUTPUT_DIR / "validation" / "stage0_detector_heldout_benchmark_report.json"
DEFAULT_COVERAGE_DOMAIN_BENCHMARK = OUTPUT_DIR / "validation" / "coverage_domain_fixture_benchmark_report.json"
DEFAULT_SCORING_SCHEMA_SEPARATION = OUTPUT_DIR / "validation" / "scoring_schema_separation_audit.json"
DEFAULT_REAL_CORPUS_STAGE0_COVERAGE_AUDIT = OUTPUT_DIR / "validation" / "real_corpus_stage0_coverage_audit.json"
DEFAULT_REDUNDANCY_HOLDOUT = OUTPUT_DIR / "validation" / "redundancy_silver_holdout_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "core_behavior_audit_v2.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "core_behavior_audit_v2.md"


def _check(name: str, passed: bool, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def _validity_checks() -> List[Dict[str, Any]]:
    scorer = CoreMetricScorer.__new__(CoreMetricScorer)
    clean = (
        "def normalize_name(value):\n"
        "    cleaned = value.strip().lower()\n"
        "    if not cleaned:\n"
        "        return 'unknown'\n"
        "    return cleaned.replace(' ', '-')\n"
    )
    corrupted = "\ufffd\ufffd <script>alert('x')</script> |||||||||| 12345 @@ @@ @@"
    short_useful = "Return True when the cache entry has already expired today."
    technical_repetition = (
        "Parameter value: retries. Parameter value: timeout. Parameter value: headers. "
        "Returns a normalized request configuration for the client."
    )
    symbol_noise = "@@@@ #### !!!! |||| 12345 67890 !!!!! #### @@@@"
    clean_gate = scorer.structural_validity_gate(clean)
    corrupted_gate = scorer.structural_validity_gate(corrupted)
    short_score = scorer.structural_validity_score(short_useful)
    technical_score = scorer.structural_validity_score(technical_repetition)
    symbol_gate = scorer.structural_validity_gate(symbol_noise)
    return [
        _check(
            "validity_passes_parseable_structured_code",
            bool(clean_gate["valid"]),
            {"score": clean_gate["score"], "violated_rules": clean_gate["details"].get("violated_rules")},
        ),
        _check(
            "validity_rejects_corrupted_markup_noise",
            not bool(corrupted_gate["valid"]),
            {"score": corrupted_gate["score"], "violated_rules": corrupted_gate["details"].get("violated_rules")},
        ),
        _check(
            "validity_keeps_short_useful_text_as_warning_not_semantic_reject",
            bool(short_score["valid"]) and not short_score["details"].get("violated_rules"),
            {
                "valid": short_score["valid"],
                "violated_rules": short_score["details"].get("violated_rules"),
                "warning_rules": short_score["details"].get("warning_rules"),
                "decision_scope": short_score["details"].get("decision_scope"),
            },
        ),
        _check(
            "validity_preserves_style_repetition_as_warning_when_structurally_usable",
            bool(technical_score["valid"]) and not technical_score["details"].get("violated_rules"),
            {
                "valid": technical_score["valid"],
                "style_bucket": technical_score["details"].get("style_bucket"),
                "violated_rules": technical_score["details"].get("violated_rules"),
                "warning_rules": technical_score["details"].get("warning_rules"),
            },
        ),
        _check(
            "validity_rejects_symbol_noise",
            not bool(symbol_gate["valid"]),
            {"score": symbol_gate["score"], "violated_rules": symbol_gate["details"].get("violated_rules")},
        ),
    ]


def _row(uid: str, text: str, *, path: str | None = None, content_type: str = "code") -> Dict[str, Any]:
    return {
        "chunk_uid": uid,
        "split": "train",
        "stage_a_pass": True,
        "bundle_id": "fixture",
        "repository_identity": "fixture/repo",
        "path": path or f"src/{uid}.py",
        "change_type": "modified",
        "content_type": content_type,
        "chunk_kind": "function",
        "text": text,
    }


def _selection_value_checks() -> List[Dict[str, Any]]:
    model = load_reference_quality_model()
    scorer = CoreMetricScorer.__new__(CoreMetricScorer)
    informative = (
        "A retry budget limits how many times a client repeats a failed request. "
        "For example, exponential backoff reduces pressure on an overloaded service "
        "because each later attempt waits longer than the previous attempt."
    )
    boilerplate = "click here sign up now terms of service privacy policy buy now " * 8
    informative_score = model.score_text(informative)["score"]
    boilerplate_score = model.score_text(boilerplate)["score"]

    base = _row(
        "base",
        "def parse(value):\n    if value is None:\n        return []\n    return value.split(',')\n",
    )
    formatted = _row(
        "formatted",
        "def parse( value ):\n\n    if value is None:\n        return [ ]\n\n    return value.split( ',' )\n",
    )
    renamed = _row(
        "renamed",
        "def decode(payload):\n    if payload is None:\n        return []\n    return payload.split(',')\n",
    )
    features = {row["chunk_uid"]: local_stage_b_features(row) for row in (base, formatted, renamed)}
    validity = scorer.structural_validity_score(informative)
    calibrated = scorer._calibrate_reference_quality(
        {"score": informative_score, "details": {"token_count": len(informative.split())}},
        validity,
        text=informative,
    )
    return [
        _check(
            "selection_value_ranks_informative_above_boilerplate",
            informative_score > boilerplate_score,
            {"informative": informative_score, "boilerplate": boilerplate_score},
        ),
        _check(
            "code_selection_value_invariant_to_formatting_and_rename",
            (
                features["base"]["code_quality_proxy"] == features["formatted"]["code_quality_proxy"]
                and features["base"]["code_quality_proxy"] == features["renamed"]["code_quality_proxy"]
            ),
            {
                "base": features["base"]["code_quality_proxy"],
                "formatted": features["formatted"]["code_quality_proxy"],
                "renamed": features["renamed"]["code_quality_proxy"],
            },
        ),
        _check(
            "selection_value_evidence_declares_no_hard_reject_authority",
            (
                calibrated["details"].get("canonical_core_axis")
                == "Selection Value Evidence"
                and calibrated["details"].get("legacy_core_axis_alias") == "Quality"
                and calibrated["details"].get("hard_reject_authority") is False
            ),
            {
                "canonical_core_axis": calibrated["details"].get("canonical_core_axis"),
                "legacy_core_axis_alias": calibrated["details"].get("legacy_core_axis_alias"),
                "construct_claim": calibrated["details"].get("construct_claim"),
                "hard_reject_authority": calibrated["details"].get("hard_reject_authority"),
            },
        ),
    ]


def _retention_policy_checks() -> List[Dict[str, Any]]:
    records = [
        _row(
            "parser",
            "def parse_items(value):\n"
            "    if value is None:\n"
            "        return []\n"
            "    return [item.strip() for item in value.split(',') if item.strip()]\n",
        ),
        _row(
            "guard",
            "def require_positive(value):\n"
            "    if value <= 0:\n"
            "        raise ValueError('value must be positive')\n"
            "    return value\n",
        ),
        _row(
            "test",
            "def test_parse_items_ignores_empty_values():\n"
            "    assert parse_items('a,, b') == ['a', 'b']\n",
            path="tests/test_parser.py",
            content_type="test",
        ),
    ]
    common = {
        "quality_weight": 0.8,
        "redundancy_weight": 0.2,
        "coverage_axes": ["bundle_id", "content_type", "path_family", "difficulty_band"],
        "minimum_exemplars": 1,
        "baseline_seed": 42,
    }
    retain_all = select_stage_b(records, budget_fraction=None, **common)
    constrained = select_stage_b(records, budget_fraction=0.55, **common)
    return [
        _check(
            "retain_all_when_no_training_budget_is_constrained",
            (
                retain_all["selection_mode"] == "retain_all"
                and len(retain_all["curated_pool"]) == len(records)
                and len(retain_all["selected"]) == len(records)
                and not retain_all["budget_not_selected"]
            ),
            {
                "selection_mode": retain_all["selection_mode"],
                "curated_pool_count": len(retain_all["curated_pool"]),
                "selected_count": len(retain_all["selected"]),
                "disposition_summary": retain_all["disposition_summary"],
            },
        ),
        _check(
            "budget_not_selected_records_remain_in_curated_pool",
            (
                len(constrained["curated_pool"]) == len(records)
                and bool(constrained["budget_not_selected"])
                and all(
                    row["curation_decision"]["curation_disposition"] == "retained"
                    and row["curation_decision"]["budget_exclusion_is_rejection"] is False
                    for row in constrained["budget_not_selected"]
                )
            ),
            {
                "selection_mode": constrained["selection_mode"],
                "curated_pool_count": len(constrained["curated_pool"]),
                "selected_count": len(constrained["selected"]),
                "budget_not_selected_count": len(constrained["budget_not_selected"]),
                "disposition_summary": constrained["disposition_summary"],
            },
        ),
    ]


def _stage_b_fixture_records(stage_b_proxy_fixtures_path: Path) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    if not stage_b_proxy_fixtures_path.exists():
        return {}, [], [f"stage_b_proxy_fixture_missing:{stage_b_proxy_fixtures_path}"]
    payload = load_json(stage_b_proxy_fixtures_path)
    records = {
        str(row["chunk_uid"]): {
            **row,
            "split": "train",
            "stage_a_pass": True,
            "bundle_id": str(row.get("bundle_id") or "fixture"),
            "repository_identity": str(row.get("repository_identity") or "fixture/repo"),
            "change_type": str(row.get("change_type") or "modified"),
            "chunk_kind": str(row.get("chunk_kind") or "fixture"),
        }
        for row in payload.get("records", [])
    }
    return records, list(payload.get("assertions") or []), []


def _stage_b_fixture_selection_value_checks(stage_b_proxy_fixtures_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    records, assertions, gaps = _stage_b_fixture_records(stage_b_proxy_fixtures_path)
    if not records:
        return [], gaps
    features = {uid: local_stage_b_features(row) for uid, row in records.items()}
    checks: List[Dict[str, Any]] = []
    for assertion in assertions:
        kind = assertion.get("type")
        if kind == "selection_value_pair":
            higher = str(assertion["higher"])
            lower = str(assertion["lower"])
            margin = float(assertion.get("minimum_margin") or 0.0)
            delta = float(features[higher]["code_quality_proxy"]) - float(features[lower]["code_quality_proxy"])
            checks.append(
                _check(
                    f"fixture:{assertion['id']}",
                    delta >= margin,
                    {
                        "higher": higher,
                        "lower": lower,
                        "higher_score": features[higher]["code_quality_proxy"],
                        "lower_score": features[lower]["code_quality_proxy"],
                        "delta": round(delta, 6),
                        "minimum_margin": margin,
                    },
                )
            )
        elif kind == "selection_value_invariance":
            left = str(assertion["left"])
            right = str(assertion["right"])
            max_delta = float(assertion.get("maximum_absolute_delta") or 0.0)
            delta = abs(float(features[left]["code_quality_proxy"]) - float(features[right]["code_quality_proxy"]))
            checks.append(
                _check(
                    f"fixture:{assertion['id']}",
                    delta <= max_delta,
                    {
                        "left": left,
                        "right": right,
                        "left_score": features[left]["code_quality_proxy"],
                        "right_score": features[right]["code_quality_proxy"],
                        "absolute_delta": round(delta, 6),
                        "maximum_absolute_delta": max_delta,
                    },
                )
            )
    return checks, gaps


def _redundancy_checks() -> List[Dict[str, Any]]:
    base = _row(
        "base",
        "def parse(value):\n    if value is None:\n        return []\n    return value.split(',')\n",
    )
    renamed = _row(
        "renamed",
        "def decode(payload):\n    if payload is None:\n        return []\n    return payload.split(',')\n",
    )
    distinct = _row(
        "distinct",
        "def total(values):\n    result = 0\n    for value in values:\n        result += value\n    return result\n",
    )
    scored = score_stage_b([base, renamed, distinct], quality_weight=0.8, redundancy_weight=0.2)
    by_id = {row["chunk_uid"]: row["stage_b_evidence"] for row in scored}
    return [
        _check(
            "redundancy_detects_structural_duplicate_after_rename",
            by_id["base"]["soft_redundancy_risk"] >= 0.85 and by_id["renamed"]["soft_redundancy_risk"] >= 0.85,
            {
                "base": by_id["base"]["soft_redundancy_risk"],
                "renamed": by_id["renamed"]["soft_redundancy_risk"],
                "structural_match_count": by_id["base"]["soft_structural_match_count"],
            },
        ),
        _check(
            "redundancy_does_not_penalize_distinct_structure",
            by_id["distinct"]["soft_structural_redundancy_risk"] == 0.0,
            {"distinct_structural_risk": by_id["distinct"]["soft_structural_redundancy_risk"]},
        ),
    ]


def _stage_b_fixture_redundancy_checks(stage_b_proxy_fixtures_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    records, assertions, gaps = _stage_b_fixture_records(stage_b_proxy_fixtures_path)
    if not records:
        return [], gaps
    scored = score_stage_b(list(records.values()), quality_weight=0.8, redundancy_weight=0.2)
    by_id = {row["chunk_uid"]: row["stage_b_evidence"] for row in scored}
    checks: List[Dict[str, Any]] = []
    for assertion in assertions:
        kind = assertion.get("type")
        if kind == "redundancy_pair_floor":
            left = str(assertion["left"])
            right = str(assertion["right"])
            floor = float(assertion.get("minimum_pair_risk") or 0.0)
            left_risk = float(by_id[left]["soft_redundancy_risk"])
            right_risk = float(by_id[right]["soft_redundancy_risk"])
            checks.append(
                _check(
                    f"fixture:{assertion['id']}",
                    min(left_risk, right_risk) >= floor,
                    {
                        "left": left,
                        "right": right,
                        "left_risk": round(left_risk, 6),
                        "right_risk": round(right_risk, 6),
                        "minimum_pair_risk": floor,
                    },
                )
            )
        elif kind == "redundancy_group":
            high_ids = [str(uid) for uid in assertion.get("higher_risk_group") or []]
            low_ids = [str(uid) for uid in assertion.get("lower_risk_group") or []]
            margin = float(assertion.get("minimum_mean_margin") or 0.0)
            high_mean = sum(float(by_id[uid]["soft_redundancy_risk"]) for uid in high_ids) / max(len(high_ids), 1)
            low_mean = sum(float(by_id[uid]["soft_redundancy_risk"]) for uid in low_ids) / max(len(low_ids), 1)
            delta = high_mean - low_mean
            checks.append(
                _check(
                    f"fixture:{assertion['id']}",
                    delta >= margin,
                    {
                        "higher_risk_group": high_ids,
                        "lower_risk_group": low_ids,
                        "higher_mean_risk": round(high_mean, 6),
                        "lower_mean_risk": round(low_mean, 6),
                        "delta": round(delta, 6),
                        "minimum_mean_margin": margin,
                    },
                )
            )
    return checks, gaps


def _coverage_checks() -> List[Dict[str, Any]]:
    original = Counter({1: 50, 2: 25, 3: 5, 4: 4})
    retained = Counter({1: 25, 2: 10, 3: 2, 4: 1})
    collapsed = Counter({1: 38})
    empty = Counter()
    retained_cov = _coverage_retention(retained, original)
    collapsed_cov = _coverage_retention(collapsed, original)
    empty_cov = _coverage_retention(empty, original)
    source_support = _distribution_bucket_support(
        Counter({"repo_a": 8, "repo_b": 2}),
        Counter({"repo_a": 5, "repo_b": 5}),
        support_scope="source_bucket",
        support_label="repository",
    )
    collapsed_support = _distribution_bucket_support(
        Counter({"repo_a": 10}),
        Counter({"repo_a": 5, "repo_b": 5}),
        support_scope="source_bucket",
        support_label="repository",
    )
    content_support = _distribution_bucket_support(
        Counter({"code": 5, "test": 3, "docs": 2}),
        Counter({"code": 5, "test": 3, "docs": 2}),
        support_scope="content_type_bucket",
        support_label="content_type",
    )
    content_collapsed = _distribution_bucket_support(
        Counter({"code": 10}),
        Counter({"code": 5, "test": 3, "docs": 2}),
        support_scope="content_type_bucket",
        support_label="content_type",
    )
    path_support = _distribution_bucket_support(
        Counter({"src/parser": 4, "tests": 3, "docs": 2, "src/network": 1}),
        Counter({"src/parser": 4, "tests": 3, "docs": 2, "src/network": 1}),
        support_scope="path_family_bucket",
        support_label="path_family",
    )
    path_collapsed = _distribution_bucket_support(
        Counter({"src/parser": 10}),
        Counter({"src/parser": 4, "tests": 3, "docs": 2, "src/network": 1}),
        support_scope="path_family_bucket",
        support_label="path_family",
    )
    return [
        _check(
            "coverage_retention_detects_tail_cluster_collapse",
            retained_cov["score"] > collapsed_cov["score"] and retained_cov["rare_cluster_retention"] > collapsed_cov["rare_cluster_retention"],
            {"retained": retained_cov, "collapsed": collapsed_cov},
        ),
        _check(
            "coverage_bucket_support_detects_source_collapse",
            source_support["retained_bucket_ratio"] > collapsed_support["retained_bucket_ratio"]
            and source_support["distribution_similarity"] > collapsed_support["distribution_similarity"],
            {"balanced": source_support, "collapsed": collapsed_support},
        ),
        _check(
            "coverage_retention_rejects_empty_selection",
            empty_cov["score"] == 0.0 and empty_cov["rare_cluster_retention"] == 0.0,
            {"empty": empty_cov},
        ),
        _check(
            "coverage_bucket_support_detects_content_type_collapse",
            content_support["retained_bucket_ratio"] > content_collapsed["retained_bucket_ratio"]
            and content_support["distribution_similarity"] > content_collapsed["distribution_similarity"],
            {"balanced": content_support, "collapsed": content_collapsed},
        ),
        _check(
            "coverage_bucket_support_detects_path_family_collapse",
            path_support["retained_bucket_ratio"] > path_collapsed["retained_bucket_ratio"]
            and path_support["distribution_similarity"] > path_collapsed["distribution_similarity"],
            {"balanced": path_support, "collapsed": path_collapsed},
        ),
    ]


def _utility_checks(selector_leakage_path: Path, confirmatory_decision_path: Path) -> tuple[List[Dict[str, Any]], List[str]]:
    gaps: List[str] = []
    checks: List[Dict[str, Any]] = []
    if selector_leakage_path.exists():
        leakage = load_json(selector_leakage_path)
        checks.append(
            _check(
                "utility_not_consumed_by_stage_b_selector_or_evidence",
                leakage.get("status") == "selector_utility_leakage_audit_passed" and not leakage.get("blockers"),
                {
                    "status": leakage.get("status"),
                    "blockers": leakage.get("blockers"),
                    "records_checked": (leakage.get("stage_b_evidence_sample") or {}).get("records_checked"),
                },
            )
        )
    else:
        gaps.append("selector_utility_leakage_audit_missing")

    if confirmatory_decision_path.exists():
        decision = load_json(confirmatory_decision_path)
        status = str(decision.get("status") or "")
        required = decision.get("required_guardrails") or decision.get("guardrails") or {}
        missing_blob = json.dumps(decision, sort_keys=True).lower()
        guardrails_complete = status == "v2_confirmatory_decision_passed"
        missing_guardrails_abstain = "abstain" in status and "missing" in missing_blob
        checks.append(
            _check(
                "utility_decision_matches_required_guardrail_state",
                guardrails_complete or missing_guardrails_abstain,
                {
                    "status": status,
                    "required_guardrails": required,
                    "accepted_states": ["passed_when_guardrails_complete", "abstain_when_guardrails_missing"],
                },
            )
        )
    else:
        gaps.append("code_domain_v2_confirmatory_decision_report_missing")
    return checks, gaps


def build(
    construct_review_path: Path,
    selector_leakage_path: Path,
    confirmatory_decision_path: Path,
    stage_b_proxy_fixtures_path: Path,
    stage0_hazard_benchmark_path: Path,
    stage0_detector_validation_path: Path,
    stage0_detector_heldout_path: Path,
    coverage_domain_benchmark_path: Path,
    scoring_schema_separation_path: Path,
    real_corpus_stage0_coverage_audit_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    fixture_selection_value_checks, fixture_selection_value_gaps = _stage_b_fixture_selection_value_checks(
        stage_b_proxy_fixtures_path
    )
    fixture_redundancy_checks, fixture_redundancy_gaps = _stage_b_fixture_redundancy_checks(stage_b_proxy_fixtures_path)
    checks_by_core = {
        "Validity": _validity_checks(),
        "Selection Value Evidence": _selection_value_checks() + fixture_selection_value_checks + _retention_policy_checks(),
        "Redundancy": _redundancy_checks() + fixture_redundancy_checks,
        "Coverage": _coverage_checks(),
    }
    utility_checks, gaps = _utility_checks(selector_leakage_path, confirmatory_decision_path)
    gaps.extend(fixture_selection_value_gaps)
    gaps.extend(fixture_redundancy_gaps)
    checks_by_core["Utility"] = utility_checks

    if construct_review_path.exists():
        construct = load_json(construct_review_path)
        checks_by_core["Selection Value Evidence"].append(
            _check(
                "intrinsic_quality_claim_rejected",
                (construct.get("decision") or {}).get("quality_as_intrinsic_core") == "rejected",
                {"decision": construct.get("decision")},
            )
        )
    else:
        gaps.append("core_construct_validity_review_missing")

    stage0_hazard_benchmark = None
    if stage0_hazard_benchmark_path.exists():
        stage0_hazard_benchmark = load_json(stage0_hazard_benchmark_path)
        if stage0_hazard_benchmark.get("status") != "stage0_hazard_fixture_benchmark_passed":
            gaps.append("stage0_hazard_fixture_benchmark_not_passing")
    else:
        gaps.append("stage0_hazard_fixture_benchmark_missing")

    stage0_detector_validation = None
    if stage0_detector_validation_path.exists():
        stage0_detector_validation = load_json(stage0_detector_validation_path)
        if stage0_detector_validation.get("status") != "stage0_detector_validation_precheck_passed_with_scope_caveats":
            gaps.append("stage0_detector_validation_precheck_not_passing")
    else:
        gaps.append("stage0_detector_validation_precheck_missing")

    stage0_detector_heldout = None
    if stage0_detector_heldout_path.exists():
        stage0_detector_heldout = load_json(stage0_detector_heldout_path)
        if stage0_detector_heldout.get("status") != "stage0_detector_heldout_benchmark_passed_with_scope_caveats":
            gaps.append("stage0_detector_heldout_benchmark_not_passing")
        gaps.append("stage0_detector_heldout_benchmark_not_external_public_benchmark")
    else:
        gaps.append("stage0_detector_heldout_benchmark_missing")

    coverage_domain_benchmark = None
    if coverage_domain_benchmark_path.exists():
        coverage_domain_benchmark = load_json(coverage_domain_benchmark_path)
        if coverage_domain_benchmark.get("status") != "coverage_domain_fixture_benchmark_passed":
            gaps.append("coverage_domain_fixture_benchmark_not_passing")
    else:
        gaps.append("coverage_domain_fixture_benchmark_missing")

    scoring_schema_separation = None
    if scoring_schema_separation_path.exists():
        scoring_schema_separation = load_json(scoring_schema_separation_path)
        if scoring_schema_separation.get("status") != "scoring_schema_separation_audit_passed":
            gaps.append("scoring_schema_separation_audit_not_passing")
    else:
        gaps.append("scoring_schema_separation_audit_missing")

    real_corpus_stage0_coverage_audit = None
    if real_corpus_stage0_coverage_audit_path.exists():
        real_corpus_stage0_coverage_audit = load_json(real_corpus_stage0_coverage_audit_path)
        if real_corpus_stage0_coverage_audit.get("status") != "real_corpus_stage0_coverage_audit_passed_with_scope_caveats":
            gaps.append("real_corpus_stage0_coverage_audit_not_passing")
        caveats = set(str(item) for item in real_corpus_stage0_coverage_audit.get("caveats") or [])
        if "true_domain_coverage_not_claimable_without_explicit_domain_metadata" in caveats:
            gaps.append("explicit_domain_metadata_missing_for_true_domain_coverage_claim")
        if "stage0_hazard_counts_do_not_replace_production_detector_validation" in caveats:
            gaps.append("real_corpus_stage0_hazard_counts_not_production_detector_validation")
    else:
        gaps.append("real_corpus_stage0_coverage_audit_missing")

    redundancy_holdout = load_json(DEFAULT_REDUNDANCY_HOLDOUT) if DEFAULT_REDUNDANCY_HOLDOUT.exists() else {}
    current_redundancy = ((redundancy_holdout.get("arm_results") or {}).get("current") or {})
    near_duplicate_hard_gate_supported = bool(current_redundancy.get("eligible_after_holdout"))

    blockers: List[str] = []
    for core, checks in checks_by_core.items():
        for row in checks:
            if not row["passed"]:
                blockers.append(f"{core}:{row['name']}")

    remaining_evidence_gaps = [
        *gaps,
        "core_behavior_fixture_suite_expanded_but_not_exhaustive",
    ]
    status = (
        "core_behavior_audit_failed"
        if blockers
        else "core_behavior_audit_development_checks_passed"
    )
    report = {
        "schema_version": "core-behavior-audit-v2",
        "status": status,
        "core_checks": checks_by_core,
        "blockers": blockers,
        "remaining_evidence_gaps": remaining_evidence_gaps,
        "metric_validity_status": "development_only_not_external_construct_validity",
        "metric_authority": {
            "structural_validity_gate": {
                "authority": "stage_a_hard_gate",
                "scope": "frozen structural usability rules",
            },
            "reference_quality_score": {
                "authority": "stage_b_selection_signal_only",
                "scope": "Selection Value Evidence; no hard-reject authority",
            },
            "exact_duplicate_indicator": {
                "authority": "stage_a_hard_gate",
                "scope": "raw or canonical-content exact duplicates with representative lineage",
            },
            "shingle_near_duplicate_indicator": {
                "authority": "stage_b_soft_signal_only",
                "hard_gate_supported": near_duplicate_hard_gate_supported,
                "scope": "fuzzy near-duplicate evidence is not an irreversible rejection rule",
            },
            "shingle_near_duplicate_risk_score": {
                "authority": "stage_b_soft_signal_only",
                "scope": "saturation and recurrence evidence",
            },
            "subset_coverage_retention_score": {
                "authority": "conditional_stage_b_stage_c_audit",
                "scope": "observable metadata axes declared by the deployment contract",
            },
            "small_lm_probe_gain_score": {
                "authority": "stage_c_validator_only",
                "scope": "frozen downstream comparison; never selector input",
            },
        },
        "decision": {
            "release_claim_supported": False,
            "core_metric_validity_fully_proven": False,
            "interpretation": (
                "Fixture behavior checks passed for the current expanded audit, "
                "but this is not a production-grade metric-validity proof. "
                "Remaining evidence gaps must be closed before claiming Core validity broadly."
            ),
        },
        "supporting_stage0_hazard_benchmark": {
            "path": str(stage0_hazard_benchmark_path),
            "status": stage0_hazard_benchmark.get("status") if isinstance(stage0_hazard_benchmark, dict) else None,
            "summary": stage0_hazard_benchmark.get("summary") if isinstance(stage0_hazard_benchmark, dict) else None,
        },
        "supporting_stage0_detector_validation": {
            "path": str(stage0_detector_validation_path),
            "status": stage0_detector_validation.get("status") if isinstance(stage0_detector_validation, dict) else None,
            "summary": stage0_detector_validation.get("summary") if isinstance(stage0_detector_validation, dict) else None,
            "axis_metrics": stage0_detector_validation.get("axis_metrics") if isinstance(stage0_detector_validation, dict) else None,
            "remaining_evidence_gaps": stage0_detector_validation.get("remaining_evidence_gaps")
            if isinstance(stage0_detector_validation, dict)
            else None,
        },
        "supporting_stage0_detector_heldout_benchmark": {
            "path": str(stage0_detector_heldout_path),
            "status": stage0_detector_heldout.get("status") if isinstance(stage0_detector_heldout, dict) else None,
            "benchmark_scope": stage0_detector_heldout.get("benchmark_scope")
            if isinstance(stage0_detector_heldout, dict)
            else None,
            "summary": stage0_detector_heldout.get("summary") if isinstance(stage0_detector_heldout, dict) else None,
            "axis_metrics": stage0_detector_heldout.get("axis_metrics") if isinstance(stage0_detector_heldout, dict) else None,
            "remaining_evidence_gaps": stage0_detector_heldout.get("remaining_evidence_gaps")
            if isinstance(stage0_detector_heldout, dict)
            else None,
        },
        "supporting_coverage_domain_benchmark": {
            "path": str(coverage_domain_benchmark_path),
            "status": coverage_domain_benchmark.get("status") if isinstance(coverage_domain_benchmark, dict) else None,
            "summary": coverage_domain_benchmark.get("summary") if isinstance(coverage_domain_benchmark, dict) else None,
        },
        "supporting_scoring_schema_separation": {
            "path": str(scoring_schema_separation_path),
            "status": scoring_schema_separation.get("status") if isinstance(scoring_schema_separation, dict) else None,
            "blockers": scoring_schema_separation.get("blockers") if isinstance(scoring_schema_separation, dict) else None,
        },
        "supporting_real_corpus_stage0_coverage_audit": {
            "path": str(real_corpus_stage0_coverage_audit_path),
            "status": real_corpus_stage0_coverage_audit.get("status")
            if isinstance(real_corpus_stage0_coverage_audit, dict)
            else None,
            "stage0": {
                "release_candidate_count": ((real_corpus_stage0_coverage_audit.get("stage0") or {}).get("release_candidate_count"))
                if isinstance(real_corpus_stage0_coverage_audit, dict)
                else None,
                "quarantined_candidate_count": (
                    (real_corpus_stage0_coverage_audit.get("stage0") or {}).get("quarantined_candidate_count")
                )
                if isinstance(real_corpus_stage0_coverage_audit, dict)
                else None,
            },
            "coverage": {
                "support_scope": ((real_corpus_stage0_coverage_audit.get("coverage") or {}).get("support_scope"))
                if isinstance(real_corpus_stage0_coverage_audit, dict)
                else None,
                "true_domain_coverage_claim_allowed": (
                    (real_corpus_stage0_coverage_audit.get("coverage") or {}).get("true_domain_coverage_claim_allowed")
                )
                if isinstance(real_corpus_stage0_coverage_audit, dict)
                else None,
            },
            "caveats": real_corpus_stage0_coverage_audit.get("caveats")
            if isinstance(real_corpus_stage0_coverage_audit, dict)
            else None,
        },
        "supporting_redundancy_holdout": {
            "path": str(DEFAULT_REDUNDANCY_HOLDOUT),
            "status": redundancy_holdout.get("status"),
            "current_eligible_after_holdout": near_duplicate_hard_gate_supported,
            "promotion_blockers": redundancy_holdout.get("promotion_blockers") or [],
        },
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Core Behavior Audit v2",
        "",
        f"Status: `{report['status']}`",
        "",
        report["decision"]["interpretation"],
        "",
        "## Core Checks",
        "",
    ]
    for core, checks in report["core_checks"].items():
        lines.append(f"### {core}")
        for row in checks:
            mark = "pass" if row["passed"] else "fail"
            lines.append(f"- `{mark}` `{row['name']}`")
        lines.append("")
    stage0 = report.get("supporting_stage0_hazard_benchmark") or {}
    lines.extend(
        [
            "## Supporting Stage-0 Hazard Benchmark",
            "",
            f"- Path: `{stage0.get('path')}`",
            f"- Status: `{stage0.get('status')}`",
        ]
    )
    summary = stage0.get("summary") if isinstance(stage0.get("summary"), dict) else {}
    if summary:
        lines.extend(
            [
                f"- Cases: `{summary.get('case_count')}`",
                f"- Passed: `{summary.get('passed_count')}`",
                f"- Failed: `{summary.get('failed_count')}`",
                "",
            ]
        )
    detector = report.get("supporting_stage0_detector_validation") or {}
    lines.extend(
        [
            "## Supporting Stage-0 Detector Validation",
            "",
            f"- Path: `{detector.get('path')}`",
            f"- Status: `{detector.get('status')}`",
            f"- Summary: `{detector.get('summary')}`",
            "",
        ]
    )
    heldout = report.get("supporting_stage0_detector_heldout_benchmark") or {}
    lines.extend(
        [
            "## Supporting Stage-0 Detector Heldout Benchmark",
            "",
            f"- Path: `{heldout.get('path')}`",
            f"- Status: `{heldout.get('status')}`",
            f"- Benchmark scope: `{heldout.get('benchmark_scope')}`",
            f"- Summary: `{heldout.get('summary')}`",
            "",
        ]
    )
    coverage_domain = report.get("supporting_coverage_domain_benchmark") or {}
    lines.extend(
        [
            "## Supporting Coverage Domain Benchmark",
            "",
            f"- Path: `{coverage_domain.get('path')}`",
            f"- Status: `{coverage_domain.get('status')}`",
        ]
    )
    coverage_summary = coverage_domain.get("summary") if isinstance(coverage_domain.get("summary"), dict) else {}
    if coverage_summary:
        lines.extend(
            [
                f"- Cases: `{coverage_summary.get('case_count')}`",
                f"- Passed: `{coverage_summary.get('passed_count')}`",
                f"- Failed: `{coverage_summary.get('failed_count')}`",
                f"- Support scopes: `{coverage_summary.get('support_scope_counts')}`",
                "",
            ]
        )
    scoring_schema = report.get("supporting_scoring_schema_separation") or {}
    lines.extend(
        [
            "## Supporting Scoring Schema Separation Audit",
            "",
            f"- Path: `{scoring_schema.get('path')}`",
            f"- Status: `{scoring_schema.get('status')}`",
            f"- Blockers: `{scoring_schema.get('blockers')}`",
            "",
        ]
    )
    real_corpus = report.get("supporting_real_corpus_stage0_coverage_audit") or {}
    lines.extend(
        [
            "## Supporting Real-Corpus Stage-0 Coverage Audit",
            "",
            f"- Path: `{real_corpus.get('path')}`",
            f"- Status: `{real_corpus.get('status')}`",
            f"- Stage-0 summary: `{real_corpus.get('stage0')}`",
            f"- Coverage summary: `{real_corpus.get('coverage')}`",
            f"- Caveats: `{real_corpus.get('caveats')}`",
            "",
        ]
    )
    lines.extend(["## Blockers", ""])
    lines.extend([f"- `{item}`" for item in report["blockers"]] or ["- None"])
    lines.extend(["", "## Remaining Evidence Gaps", ""])
    lines.extend([f"- `{item}`" for item in report["remaining_evidence_gaps"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Core behavior audit v2.")
    parser.add_argument("--construct-review", type=Path, default=DEFAULT_CONSTRUCT_REVIEW)
    parser.add_argument("--selector-leakage", type=Path, default=DEFAULT_SELECTOR_LEAKAGE)
    parser.add_argument("--confirmatory-decision", type=Path, default=DEFAULT_CONFIRMATORY_DECISION)
    parser.add_argument("--stage-b-proxy-fixtures", type=Path, default=DEFAULT_STAGE_B_PROXY_FIXTURES)
    parser.add_argument("--stage0-hazard-benchmark", type=Path, default=DEFAULT_STAGE0_HAZARD_BENCHMARK)
    parser.add_argument("--stage0-detector-validation", type=Path, default=DEFAULT_STAGE0_DETECTOR_VALIDATION)
    parser.add_argument("--stage0-detector-heldout", type=Path, default=DEFAULT_STAGE0_DETECTOR_HELDOUT)
    parser.add_argument("--coverage-domain-benchmark", type=Path, default=DEFAULT_COVERAGE_DOMAIN_BENCHMARK)
    parser.add_argument("--scoring-schema-separation", type=Path, default=DEFAULT_SCORING_SCHEMA_SEPARATION)
    parser.add_argument("--real-corpus-stage0-coverage-audit", type=Path, default=DEFAULT_REAL_CORPUS_STAGE0_COVERAGE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(
        args.construct_review,
        args.selector_leakage,
        args.confirmatory_decision,
        args.stage_b_proxy_fixtures,
        args.stage0_hazard_benchmark,
        args.stage0_detector_validation,
        args.stage0_detector_heldout,
        args.coverage_domain_benchmark,
        args.scoring_schema_separation,
        args.real_corpus_stage0_coverage_audit,
        args.output,
        args.md_output,
    )
    print({"status": report["status"], "blockers": report["blockers"], "remaining_evidence_gaps": report["remaining_evidence_gaps"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
