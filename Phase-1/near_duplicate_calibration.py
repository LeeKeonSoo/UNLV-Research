from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final, Literal, TypeAlias, assert_never

from pydantic import BaseModel, ConfigDict, Field

from redundancy_v2 import RedundancySettings, RedundancyUnit, RelationType, classify_relation, tokenize

Domain = Literal["code", "math", "general"]
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMap: TypeAlias = dict[str, JsonValue]
DEFAULT_CONFIG: Final = "configs/near_duplicate_calibration_v1.json"


@dataclass(frozen=True, slots=True)
class NearCalibrationError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


class NearSettingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    near_minimum_tokens: int = Field(ge=2)
    maximum_changed_ratio: float = Field(gt=0.0, lt=1.0)
    maximum_changed_tokens: int = Field(ge=1)


class AcceptanceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    require_all_eligible_positives: Literal[True]
    require_zero_semantic_false_positives: Literal[True]
    require_zero_short_positive_removals: Literal[True]


class SelectorBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_outcomes_available: Literal[False]
    utility_available: Literal[False]
    runtime_activation_mutation_allowed: Literal[False]


class NearCalibrationProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["near-duplicate-calibration-protocol-v1"]
    status: Literal["block_10a_frozen_before_calibration"]
    domains: tuple[Domain, ...]
    target_token_counts: tuple[int, ...] = Field(min_length=2)
    claimed_minimum_tokens: int = Field(ge=2)
    candidate_near_minimum_tokens: tuple[int, ...] = Field(min_length=1)
    candidate_maximum_changed_ratios: tuple[float, ...] = Field(min_length=1)
    candidate_maximum_changed_tokens: tuple[int, ...] = Field(min_length=1)
    current_setting: NearSettingSpec
    positive_operators: dict[Domain, str]
    hard_negative_operators: dict[Domain, str]
    acceptance: AcceptanceSpec
    selector_boundary: SelectorBoundary
    claim_boundary: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class PairCase:
    case_id: str
    domain: Domain
    target_token_count: int
    base_text: str
    positive_text: str
    hard_negative_text: str
    positive_witness: str
    hard_negative_witness: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pad_general(target: int) -> str:
    prefix = "The measured process increases the observed result because evidence supports the conclusion"
    words = prefix.split()
    words.extend(f"context{index}" for index in range(max(0, target - len(words))))
    return " ".join(words) + "."


def _pad_code(target: int) -> str:
    base = "def compute(values):\n    total = 0\n    for item in values:\n        total = total + item\n    return total"
    missing = target - len(tokenize(base))
    return base if missing <= 1 else base + "\n# " + " ".join(f"context{index}" for index in range(missing - 1))


def _pad_math(target: int) -> str:
    expression = "alpha * (beta + gamma)"
    while len(tokenize(expression)) < target:
        expression += f" + term{len(tokenize(expression))}"
    return expression


def _fixture(domain: Domain, target: int) -> PairCase:
    match domain:
        case "code":
            base = _pad_code(target)
            positive = base.replace("return total", "return total;", 1)
            negative = base.replace("return total", "return total,", 1)
            positive_verified = ast.dump(ast.parse(base)) == ast.dump(ast.parse(positive))
            negative_verified = ast.dump(ast.parse(base)) != ast.dump(ast.parse(negative))
        case "math":
            base = _pad_math(target)
            positive = base + ";"
            negative = base + ","
            positive_verified = ast.dump(ast.parse(base)) == ast.dump(ast.parse(positive))
            negative_verified = ast.dump(ast.parse(base)) != ast.dump(ast.parse(negative))
        case "general":
            base = _pad_general(target)
            positive = base.replace(" because ", " because\n", 1)
            negative = base.replace(" increases ", " decreases ", 1)
            positive_verified = tokenize(base) == tokenize(positive)
            negative_verified = tokenize(base) != tokenize(negative)
        case unreachable:
            assert_never(unreachable)
    if not positive_verified or not negative_verified:
        raise NearCalibrationError(f"near_fixture_witness_failed:{domain}:{target}")
    return PairCase(
        case_id=f"{domain}-{target}",
        domain=domain,
        target_token_count=target,
        base_text=base,
        positive_text=positive,
        hard_negative_text=negative,
        positive_witness="mechanically_verified_equivalent",
        hard_negative_witness="mechanically_verified_semantic_structure_change",
    )


def _settings(spec: NearSettingSpec) -> RedundancySettings:
    return RedundancySettings(
        short_exact_only_max_tokens=spec.near_minimum_tokens - 1,
        near_min_tokens=spec.near_minimum_tokens,
        near_max_changed_ratio=spec.maximum_changed_ratio,
        near_max_changed_tokens=spec.maximum_changed_tokens,
    )


def _is_near(case: PairCase, text: str, setting: NearSettingSpec) -> bool:
    relation = classify_relation(
        RedundancyUnit(f"{case.case_id}-base", case.base_text),
        RedundancyUnit(f"{case.case_id}-candidate", text),
        _settings(setting),
    )
    return relation.relation is RelationType.NEAR_SUBSTITUTE


def _setting_report(cases: tuple[PairCase, ...], setting: NearSettingSpec, claimed_minimum: int) -> JsonMap:
    eligible = tuple(case for case in cases if case.target_token_count >= claimed_minimum)
    short = tuple(case for case in cases if case.target_token_count < claimed_minimum)
    positive_hits = sum(_is_near(case, case.positive_text, setting) for case in eligible)
    semantic_false_positives = sum(_is_near(case, case.hard_negative_text, setting) for case in cases)
    short_positive_hits = sum(_is_near(case, case.positive_text, setting) for case in short)
    passed = positive_hits == len(eligible) and semantic_false_positives == 0 and short_positive_hits == 0
    return {
        "near_minimum_tokens": setting.near_minimum_tokens,
        "maximum_changed_ratio": setting.maximum_changed_ratio,
        "maximum_changed_tokens": setting.maximum_changed_tokens,
        "eligible_positive_hits": positive_hits,
        "eligible_positive_total": len(eligible),
        "semantic_false_positive_count": semantic_false_positives,
        "short_positive_removal_count": short_positive_hits,
        "passed": passed,
    }


def _report_hash(report: JsonMap) -> str:
    encoded = json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def build_near_duplicate_calibration(root: Path) -> JsonMap:
    protocol_path = root / DEFAULT_CONFIG
    protocol = NearCalibrationProtocol.model_validate_json(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol.domains) != ("code", "math", "general"):
        raise NearCalibrationError("near_calibration_domain_matrix_invalid")
    if set(protocol.positive_operators) != set(protocol.domains) or set(
        protocol.hard_negative_operators
    ) != set(protocol.domains):
        raise NearCalibrationError("near_calibration_operator_matrix_invalid")
    cases = tuple(_fixture(domain, target) for domain in protocol.domains for target in protocol.target_token_counts)
    candidates = tuple(
        NearSettingSpec(
            near_minimum_tokens=minimum,
            maximum_changed_ratio=ratio,
            maximum_changed_tokens=changed,
        )
        for minimum, ratio, changed in product(
            protocol.candidate_near_minimum_tokens,
            protocol.candidate_maximum_changed_ratios,
            protocol.candidate_maximum_changed_tokens,
        )
    )
    setting_reports = tuple(_setting_report(cases, setting, protocol.claimed_minimum_tokens) for setting in candidates)
    current = _setting_report(cases, protocol.current_setting, protocol.claimed_minimum_tokens)
    eligible_cases = tuple(case for case in cases if case.target_token_count >= protocol.claimed_minimum_tokens)
    collision_cases = tuple(
        case
        for case in eligible_cases
        if _is_near(case, case.positive_text, protocol.current_setting)
        and _is_near(case, case.hard_negative_text, protocol.current_setting)
    )
    collision_domains = sorted({case.domain for case in collision_cases})
    positive_miss_domains = sorted(
        {
            case.domain
            for case in eligible_cases
            if not _is_near(case, case.positive_text, protocol.current_setting)
        }
    )
    fixture_projection: list[JsonValue] = [
        {
            "case_id": case.case_id,
            "base_sha256": hashlib.sha256(case.base_text.encode()).hexdigest(),
            "positive_sha256": hashlib.sha256(case.positive_text.encode()).hexdigest(),
            "hard_negative_sha256": hashlib.sha256(case.hard_negative_text.encode()).hexdigest(),
            "positive_witness": case.positive_witness,
            "hard_negative_witness": case.hard_negative_witness,
        }
        for case in cases
    ]
    report: JsonMap = {
        "schema_version": "near-duplicate-calibration-v1",
        "status": "blocked_threshold_not_identifiable",
        "protocol_sha256": _sha256(protocol_path),
        "domains": list(protocol.domains),
        "target_token_counts": list(protocol.target_token_counts),
        "positive_case_count": len(cases),
        "hard_negative_case_count": len(cases),
        "eligible_positive_case_count": sum(case.target_token_count >= protocol.claimed_minimum_tokens for case in cases),
        "candidate_setting_count": len(setting_reports),
        "passing_setting_count": sum(bool(item["passed"]) for item in setting_reports),
        "semantic_false_positive_count": current["semantic_false_positive_count"],
        "feature_collision_count": len(collision_cases),
        "collision_domains": collision_domains,
        "positive_miss_domains": positive_miss_domains,
        "fixture_manifest_sha256": _report_hash(
            {
                "cases": fixture_projection,
                "positive_operators": dict(protocol.positive_operators),
                "hard_negative_operators": dict(protocol.hard_negative_operators),
            }
        ),
        "setting_reports": list(setting_reports),
        "threshold_emitted": False,
        "operating_point_decisions": [
            {"profile_id": "normal", "status": "blocked", "threshold_emitted": False},
            {"profile_id": "hard", "status": "blocked", "threshold_emitted": False},
        ],
        "same_policy_family": True,
        "hard_subset_of_normal_required": True,
        "safe_family_edge_authorized": False,
        "runtime_activation_mutated": False,
        "benchmark_outcomes_read": False,
        "utility_read": False,
        "recommended_disposition": "require_external_equivalence_witness",
        "claim_boundary": protocol.claim_boundary,
    }
    report["report_sha256"] = _report_hash(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Block 10A near-duplicate calibration report.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_near_duplicate_calibration(arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passing_setting_count": report["passing_setting_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
