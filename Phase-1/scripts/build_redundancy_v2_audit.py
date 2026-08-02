# /// script
# requires-python = ">=3.12"
# dependencies = ["pydantic>=2.10"]
# ///
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redundancy_v2 import RedundancySettings, RedundancyUnit
from redundancy_v2_audit import RedundancyAuditCase, build_redundancy_audit


class FixtureCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: str
    left: str
    right: str
    expected_relation: str
    semantic_candidate: bool = False


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    metamorphic_equivalence_payloads: tuple[str, ...]
    cases: tuple[FixtureCase, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the typed cross-domain Redundancy v2 behavior audit.")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "validation" / "fixtures" / "redundancy_v2_cases.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "redundancy_v2_behavior_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FixtureBundle.model_validate_json(args.fixtures.read_text(encoding="utf-8"))
    cases = [
        RedundancyAuditCase(
            case_id=case.id,
            role=case.role,
            left=RedundancyUnit(f"{case.id}:left", case.left),
            right=RedundancyUnit(f"{case.id}:right", case.right),
            expected_relation=case.expected_relation,
            semantic_candidate=case.semantic_candidate,
        )
        for case in bundle.cases
    ]
    for index, text in enumerate(bundle.metamorphic_equivalence_payloads):
        formatting_variant = text.replace("\n", "\r\n") if "\n" in text else text + "\r\n"
        cases.extend(
            (
                RedundancyAuditCase(
                    case_id=f"metamorphic_exact_{index}",
                    role="safe_family_positive",
                    left=RedundancyUnit(f"metamorphic_exact_{index}:left", text),
                    right=RedundancyUnit(f"metamorphic_exact_{index}:right", text),
                    expected_relation="exact_equivalent",
                ),
                RedundancyAuditCase(
                    case_id=f"metamorphic_formatting_{index}",
                    role="safe_family_positive",
                    left=RedundancyUnit(f"metamorphic_formatting_{index}:left", text),
                    right=RedundancyUnit(f"metamorphic_formatting_{index}:right", formatting_variant),
                    expected_relation="formatting_equivalent",
                ),
            )
        )
    report = build_redundancy_audit(tuple(cases), RedundancySettings(), confidence_level=0.95)
    payload = {
        "schema_version": "redundancy-v2-behavior-audit-v1",
        "fixture_schema_version": bundle.schema_version,
        **asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[redundancy-v2-audit] passed={report.passed} safe={report.safe_family_positive_count} "
        f"retain={report.retain_control_count} candidate={report.candidate_only_count} output={args.output}"
    )


if __name__ == "__main__":
    main()
