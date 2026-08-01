#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from data_eval_common import OUTPUT_DIR, save_json


ROOT: Final = Path(__file__).resolve().parent
CONTRACT_PATH: Final = ROOT / "configs" / "canonical_execution_path_v1.json"
REPORT_PATH: Final = OUTPUT_DIR / "validation" / "canonical_paper_evidence_run_report.json"


@dataclass(frozen=True, slots=True)
class CanonicalStep:
    script: str
    role: str


def _load_steps() -> tuple[CanonicalStep, ...]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    raw_steps = payload["canonical_execution_path"]
    steps = tuple(
        CanonicalStep(script=str(entry["script"]), role=str(entry["role"]))
        for entry in raw_steps
    )
    if not steps:
        raise RuntimeError("canonical execution contract has no steps")
    for step in steps:
        if Path(step.script).name != step.script or not (ROOT / step.script).is_file():
            raise RuntimeError(f"invalid canonical script: {step.script}")
    return steps


def _run_step(step: CanonicalStep) -> dict[str, str | int | bool]:
    completed = subprocess.run(
        [sys.executable, step.script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    decision_exit = completed.returncode == 2
    succeeded = completed.returncode == 0 or decision_exit
    return {
        "script": step.script,
        "role": step.role,
        "returncode": completed.returncode,
        "succeeded": succeeded,
        "decision_blocked": decision_exit,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _render_plan(steps: tuple[CanonicalStep, ...]) -> str:
    lines = ["Canonical paper-evidence rebuild:"]
    lines.extend(f"  {index}. {step.script} - {step.role}" for index, step in enumerate(steps, start=1))
    lines.append("Decision exit code 2 is recorded as blocked evidence, not a runner crash.")
    return "\n".join(lines)


def _report_status(results: list[dict[str, str | int | bool]]) -> str:
    if any(result["succeeded"] is False for result in results):
        return "canonical_paper_evidence_run_failed"
    if any(result["decision_blocked"] is True for result in results):
        return "canonical_paper_evidence_run_blocked"
    return "canonical_paper_evidence_run_passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run every manifest-defined rebuild step.")
    args = parser.parse_args()
    steps = _load_steps()
    if not args.execute:
        print(_render_plan(steps))
        print("Re-run with --execute to rebuild reports.")
        return 0

    results: list[dict[str, str | int | bool]] = []
    for step in steps:
        print(f"[canonical-evidence] {step.script}", flush=True)
        result = _run_step(step)
        results.append(result)
        if result["succeeded"] is False:
            break

    status = _report_status(results)
    report = {
        "schema_version": "canonical-paper-evidence-run-v1",
        "status": status,
        "contract": str(CONTRACT_PATH.relative_to(ROOT)),
        "utility_scope": "Stage C validation only; never selector objective",
        "results": results,
    }
    save_json(REPORT_PATH, report)
    print(f"[canonical-evidence] {status}")
    if status == "canonical_paper_evidence_run_failed":
        return 1
    if status == "canonical_paper_evidence_run_blocked":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
