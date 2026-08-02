#!/usr/bin/env python3
# Run: python scripts/build_development_selection_v1_audit.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from development_selection import load_development_protocol
from development_selection_preflight import CurrentDevelopmentPreflight, evaluate_current_development_preflight


DEFAULT_OUTPUT = ROOT / "validation" / "frozen_contracts" / "development_selection_v1_current_preflight.json"


class DevelopmentSelectionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["development-selection-v1-audit"]
    status: str
    contract_mechanics_test: str
    contract_fixture_only_not_empirical_evidence: Literal[True]
    profiles_frozen: Literal[False]
    normal_selection_rule: str
    hard_selection_rule: str
    benchmark_outcomes_available_to_selector: Literal[False]
    confirmatory_outcomes_available_to_selector: Literal[False]
    current_repository_preflight: CurrentDevelopmentPreflight
    interpretation: str


def build_audit(root: Path) -> DevelopmentSelectionAudit:
    protocol = load_development_protocol(root / "configs" / "development_selection_v1.json")
    current = evaluate_current_development_preflight(root)
    return DevelopmentSelectionAudit(
        schema_version="development-selection-v1-audit",
        status=current.status.value,
        contract_mechanics_test="validation/test_development_selection_v1.py",
        contract_fixture_only_not_empirical_evidence=True,
        profiles_frozen=False,
        normal_selection_rule=protocol.normal_selection_rule,
        hard_selection_rule=protocol.hard_selection_rule,
        benchmark_outcomes_available_to_selector=False,
        confirmatory_outcomes_available_to_selector=False,
        current_repository_preflight=current,
        interpretation="Block 8 cannot freeze Normal or Hard until every listed empirical blocker is closed.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Block 8 current-evidence preflight audit.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output: Path = args.output
    payload = build_audit(ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    blocker_count = len(payload.current_repository_preflight.blocker_codes)
    print(f"[development-selection-v1] status={payload.status} blockers={blocker_count} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
