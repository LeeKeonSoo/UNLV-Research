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

from validity_v2 import TextField, ValidityInput
from validity_v2_audit import ValidityAuditCase, build_validity_audit


class FixtureCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    role: str
    fields: dict[str, str]
    source_record_text: str | None = None
    expected_status: str
    expected_action: str
    expected_reason: str | None = None


class FixtureBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    cases: tuple[FixtureCase, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cross-domain Validity v2 behavior audit.")
    parser.add_argument("--fixtures", type=Path, default=ROOT / "validation" / "fixtures" / "validity_v2_cases.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "validation" / "frozen_contracts" / "validity_v2_behavior_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = FixtureBundle.model_validate_json(args.fixtures.read_text(encoding="utf-8"))
    cases = tuple(
        ValidityAuditCase(
            case_id=case.id,
            role=case.role,
            input_unit=ValidityInput(
                text_fields=tuple(TextField(name, text) for name, text in case.fields.items()),
                source_record_text=case.source_record_text,
            ),
            expected_status=case.expected_status,
            expected_action=case.expected_action,
            expected_reason=case.expected_reason,
        )
        for case in bundle.cases
    )
    report = build_validity_audit(cases, confidence_level=0.95)
    payload = {
        "schema_version": "validity-v2-behavior-audit-v1",
        "fixture_schema_version": bundle.schema_version,
        **asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(
        f"[validity-v2-audit] passed={report.passed} clean={report.clean_control_count} "
        f"positive={report.positive_count} output={args.output}"
    )


if __name__ == "__main__":
    main()
