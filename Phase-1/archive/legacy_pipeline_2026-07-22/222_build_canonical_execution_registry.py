#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
ROOT: Final = Path(__file__).resolve().parent
CONTRACT_PATH: Final = ROOT / "configs" / "canonical_execution_path_v1.json"
REPORT_PATH: Final = OUTPUT_DIR / "validation" / "canonical_execution_registry_report.json"
MD_REPORT_PATH: Final = OUTPUT_DIR / "validation" / "canonical_execution_registry_report.md"
NUMBERED_SCRIPT_PATTERN: Final = re.compile(r"^\d+_.+\.py$")


def _exists_with_hash(path: Path) -> JsonMap:
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _numbered_scripts() -> list[str]:
    return sorted(path.name for path in ROOT.glob("*.py") if NUMBERED_SCRIPT_PATTERN.match(path.name))


def _forbidden_hits(script_names: list[str], forbidden_fragments: list[str]) -> list[JsonMap]:
    hits = []
    for script in script_names:
        lowered = script.lower()
        matched = [fragment for fragment in forbidden_fragments if fragment.lower() in lowered]
        if matched:
            hits.append({"script": script, "matched_fragments": matched})
    return hits


def _canonical_entries(contract: JsonMap) -> list[JsonMap]:
    entries = []
    for entry in contract["canonical_execution_path"]:
        script_path = ROOT / entry["script"]
        output_paths = [ROOT / output for output in entry["expected_outputs"]]
        entries.append(
            {
                "script": entry["script"],
                "command": entry["command"],
                "role": entry["role"],
                "script_source": _exists_with_hash(script_path),
                "expected_outputs": [_exists_with_hash(path) for path in output_paths],
            }
        )
    return entries


def _support_reports(contract: JsonMap) -> list[JsonMap]:
    reports = []
    for report in contract["support_reports"]:
        path = ROOT / report["path"]
        reports.append({"path": report["path"], "role": report["role"], "source": _exists_with_hash(path)})
    return reports


def _missing_expected_outputs(entries: list[JsonMap]) -> list[str]:
    missing = []
    for entry in entries:
        for output in entry["expected_outputs"]:
            if output["exists"] is not True:
                missing.append(f"{entry['script']}::{output['path']}")
    return missing


def _missing_support_reports(reports: list[JsonMap]) -> list[str]:
    return [report["path"] for report in reports if report["source"]["exists"] is not True]


def _render_markdown(report: JsonMap) -> str:
    lines = [
        "# Canonical Execution Registry",
        "",
        f"Status: `{report['status']}`",
        f"Scope: `{report['scope']}`",
        "",
        "## Canonical Path",
        "",
        f"Runner: `{report['canonical_runner']['command']}`",
        f"Active framework entry points: `{', '.join(report['active_surface']['active_entry_points'])}`",
        f"Compatibility entry points: `{', '.join(report['active_surface']['compatibility_entry_points'])}`",
        "",
        "| Step | Script | Role |",
        "| ---: | --- | --- |",
    ]
    for index, entry in enumerate(report["canonical_execution_path"], start=1):
        lines.append(f"| {index} | `{entry['script']}` | {entry['role']} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Canonical scripts: `{report['summary']['canonical_count']}`",
            f"- Support reports: `{report['summary']['support_report_count']}`",
            f"- Historical numbered scripts: `{report['summary']['historical_numbered_script_count']}`",
            f"- Missing expected outputs: `{report['missing_expected_outputs']}`",
            f"- Missing support reports: `{report['missing_support_reports']}`",
            "",
            "## Claim Boundary",
            "",
            report["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def build() -> JsonMap:
    contract = load_json(CONTRACT_PATH)
    runner = contract["canonical_runner"]
    runner_source = _exists_with_hash(ROOT / runner["script"])
    numbered_scripts = _numbered_scripts()
    canonical_entries = _canonical_entries(contract)
    support_reports = _support_reports(contract)
    canonical_scripts = [entry["script"] for entry in canonical_entries]
    missing_scripts = [entry["script"] for entry in canonical_entries if entry["script_source"]["exists"] is not True]
    missing_outputs = _missing_expected_outputs(canonical_entries)
    missing_support = _missing_support_reports(support_reports)
    forbidden_hits = _forbidden_hits(canonical_scripts, contract["forbidden_script_name_fragments"])
    active_entry_points = [str(item) for item in contract["active_entry_points"]]
    compatibility_entry_points = [str(item) for item in contract["compatibility_entry_points"]]
    managed_scripts = set(canonical_scripts) | set(active_entry_points) | set(compatibility_entry_points)
    historical_scripts = [script for script in numbered_scripts if script not in managed_scripts]
    blockers = (
        ([] if runner_source["exists"] is True else [str(runner["script"])])
        + missing_scripts
        + missing_outputs
        + missing_support
        + [hit["script"] for hit in forbidden_hits]
    )
    report = {
        "schema_version": "canonical-execution-registry-v1",
        "status": "canonical_execution_registry_passed" if not blockers else "canonical_execution_registry_blocked",
        "scope": contract["scope"],
        "claim_boundary": contract["claim_boundary"],
        "canonical_path_is_lightweight_rebuild": True,
        "environment": contract["environment"],
        "canonical_runner": {**runner, "source": runner_source},
        "summary": {
            "canonical_count": len(canonical_entries),
            "support_report_count": len(support_reports),
            "numbered_script_count": len(numbered_scripts),
            "historical_numbered_script_count": len(historical_scripts),
        },
        "canonical_execution_path": canonical_entries,
        "active_surface": {
            "active_entry_points": active_entry_points,
            "compatibility_entry_points": compatibility_entry_points,
            "historical_scripts_are_not_active_entry_points": True,
        },
        "support_reports": support_reports,
        "historical_or_experimental_scripts": historical_scripts,
        "missing_canonical_scripts": missing_scripts,
        "missing_expected_outputs": missing_outputs,
        "missing_support_reports": missing_support,
        "forbidden_canonical_script_hits": forbidden_hits,
        "forbidden_in_canonical_path": contract["forbidden_in_canonical_path"],
        "utility_scope": contract["utility_scope"],
    }
    save_json(REPORT_PATH, report)
    MD_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    return 0 if report["status"] == "canonical_execution_registry_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
