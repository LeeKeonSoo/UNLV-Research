#!/usr/bin/env python3
"""Audit Core-vs-diagnostic metric separation for scoring artifacts."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from data_eval_common import CORE_SELECTION_METRICS, DIAGNOSTIC_METRICS, OUTPUT_DIR, save_json


DEFAULT_SCORER_SCRIPT = Path("03_score_core_metrics.py")
DEFAULT_SELECTOR_LEAKAGE = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "scoring_schema_separation_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "scoring_schema_separation_audit.md"
FORBIDDEN_CORE_TERMS = {
    "predictive_utility_proxy",
    "diagnostic_predictive_utility",
    "small_lm_probe_gain_score",
    "fixed_token_probe_gain_score",
    "utility_probe",
}


def _load_scoring_module(path: Path):
    spec = importlib.util.spec_from_file_location("score_core_metrics_module", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_uses_grouped_metric_contract(path: Path) -> Dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function_names: Set[str] = set()
    call_names: Set[str] = set()
    method_call_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names.add(node.name)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                call_names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                method_call_names.add(fn.attr)
    return {
        "split_metric_groups_defined": "split_metric_groups" in function_names,
        "split_metric_groups_called": "split_metric_groups" in call_names,
        "grouped_scorer_api_called": "score_chunks_grouped" in method_call_names
        or "score_chunk_grouped" in method_call_names,
    }


def _sample_split(module: Any) -> Dict[str, Any]:
    sample_payload = {"score": 0.0, "details": {"fixture": True}}
    metrics = {
        name: {"score": idx / 100.0, "details": {"metric": name}}
        for idx, name in enumerate((*CORE_SELECTION_METRICS, *DIAGNOSTIC_METRICS), start=1)
    }
    # Include a few noncanonical extras to make sure the splitter does not
    # accidentally promote unknown raw-scorer fields.
    metrics["noncanonical_extra_metric"] = sample_payload
    core, diagnostic = module.split_metric_groups(metrics)
    return {
        "core_metric_keys": sorted(core),
        "diagnostic_metric_keys": sorted(diagnostic),
        "forbidden_core_terms_seen": sorted(set(core).intersection(FORBIDDEN_CORE_TERMS)),
        "predictive_utility_in_diagnostic": "predictive_utility_proxy" in diagnostic,
        "extra_metric_promoted": "noncanonical_extra_metric" in core or "noncanonical_extra_metric" in diagnostic,
    }


def _read_selector_leakage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "status": None, "blockers": ["selector_utility_leakage_audit_missing"]}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "status": payload.get("status"),
        "blockers": payload.get("blockers") or [],
    }


def build(
    scorer_script_path: Path,
    selector_leakage_path: Path,
    output_path: Path,
    md_output_path: Path,
) -> Dict[str, Any]:
    blockers: List[str] = []
    constants = {
        "core_selection_metrics": list(CORE_SELECTION_METRICS),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
        "forbidden_core_terms": sorted(FORBIDDEN_CORE_TERMS),
        "forbidden_core_terms_in_core_constants": sorted(set(CORE_SELECTION_METRICS).intersection(FORBIDDEN_CORE_TERMS)),
        "predictive_utility_in_diagnostic_constants": "predictive_utility_proxy" in set(DIAGNOSTIC_METRICS),
    }
    if constants["forbidden_core_terms_in_core_constants"]:
        blockers.append("forbidden_terms_in_core_selection_constants")
    if not constants["predictive_utility_in_diagnostic_constants"]:
        blockers.append("predictive_utility_proxy_not_diagnostic_constant")

    source_contract = _source_uses_grouped_metric_contract(scorer_script_path)
    if not source_contract["split_metric_groups_defined"]:
        blockers.append("split_metric_groups_not_defined")
    if not source_contract["split_metric_groups_called"] and not source_contract["grouped_scorer_api_called"]:
        blockers.append("no_metric_grouping_boundary_called")

    module = _load_scoring_module(scorer_script_path)
    split_contract = _sample_split(module)
    if split_contract["forbidden_core_terms_seen"]:
        blockers.append("split_contract_places_forbidden_terms_in_core")
    if not split_contract["predictive_utility_in_diagnostic"]:
        blockers.append("split_contract_missing_predictive_utility_diagnostic")
    if split_contract["extra_metric_promoted"]:
        blockers.append("split_contract_promotes_unknown_extra_metric")

    leakage = _read_selector_leakage(selector_leakage_path)
    if leakage["status"] != "selector_utility_leakage_audit_passed" or leakage["blockers"]:
        blockers.append("selector_utility_leakage_audit_not_passing")

    report = {
        "schema_version": "scoring-schema-separation-audit-v1",
        "status": "scoring_schema_separation_audit_passed" if not blockers else "scoring_schema_separation_audit_failed",
        "constants": constants,
        "source_contract": source_contract,
        "split_contract": split_contract,
        "selector_utility_leakage_audit": leakage,
        "blockers": blockers,
        "interpretation": (
            "Raw compatibility scorer methods may compute diagnostic fields, but canonical "
            "scoring paths must call the grouped metric boundary and place Utility surrogates "
            "only under diagnostic_metrics, away from core_metrics and Stage-B evidence."
        ),
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Scoring Schema Separation Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        report["interpretation"],
        "",
        "## Contracts",
        "",
        f"- Forbidden Core terms in constants: `{report['constants']['forbidden_core_terms_in_core_constants']}`",
        f"- Predictive Utility in diagnostics: `{report['constants']['predictive_utility_in_diagnostic_constants']}`",
        f"- Split helper defined: `{report['source_contract']['split_metric_groups_defined']}`",
        f"- Split helper called: `{report['source_contract']['split_metric_groups_called']}`",
        f"- Grouped scorer API called: `{report['source_contract']['grouped_scorer_api_called']}`",
        f"- Split forbidden Core terms seen: `{report['split_contract']['forbidden_core_terms_seen']}`",
        f"- Selector leakage audit status: `{report['selector_utility_leakage_audit']['status']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend([f"- `{blocker}`" for blocker in report["blockers"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scoring schema separation audit.")
    parser.add_argument("--scorer-script", type=Path, default=DEFAULT_SCORER_SCRIPT)
    parser.add_argument("--selector-leakage", type=Path, default=DEFAULT_SELECTOR_LEAKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    report = build(args.scorer_script, args.selector_leakage, args.output, args.md_output)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
