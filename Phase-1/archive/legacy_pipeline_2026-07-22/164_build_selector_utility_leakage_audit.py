#!/usr/bin/env python3
"""Audit that Utility surrogates are not consumed by Stage-B selectors."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from data_eval_common import OUTPUT_DIR, save_json, sha256_file


DEFAULT_POLICY = Path("policy") / "subsets.py"
DEFAULT_TEMPORAL_CODE_SELECTOR = Path("ingestion") / "code_selection.py"
DEFAULT_STAGE_B_SCORED = OUTPUT_DIR / "temporal_code_collection" / "stage_b_code_domain_v2" / "train_scored_full_selector.jsonl"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "selector_utility_leakage_audit.md"

FORBIDDEN_TERMS = {
    "predictive_utility_proxy",
    "diagnostic_predictive_utility",
    "small_lm_probe_gain_score",
    "fixed_token_probe_gain_score",
    "utility_probe",
}
SELECTOR_FUNCTIONS = {
    "_axis_scores",
    "_objective_components",
    "_selection_score",
    "_passes_stage_b",
    "_selector_config",
    "_stage_b_rank",
}
TEMPORAL_CODE_SELECTOR_FUNCTIONS = {
    "local_stage_b_features",
    "score_stage_b",
    "select_stage_b",
    "_redundancy_evidence",
    "_structural_saturation_risk",
}
TEMPORAL_CODE_STAGE_B_EVIDENCE_ALLOWLIST = {
    "ast_node_count",
    "code_quality_proxy",
    "coverage_buckets",
    "length_support",
    "lexical_candidate_count",
    "lexical_or_identifier_diversity",
    "pass_through_assignment_ratio",
    "redundancy_search_mode",
    "semantic_token_proxy_count",
    "soft_lexical_redundancy_risk",
    "soft_redundancy_risk",
    "soft_redundancy_support",
    "soft_structural_match_count",
    "soft_structural_redundancy_risk",
    "stage_b_objective_score",
    "structural_richness",
    "token_proxy_count",
}


def _function_source_names(policy_path: Path, function_name: str) -> Set[str]:
    tree = ast.parse(policy_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            names: Set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    names.add(child.id)
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    names.add(child.value)
            return names
    return set()


def _read_stage_b_fields(path: Path, limit: int | None = None) -> Dict[str, Any]:
    stage_b_keys: Set[str] = set()
    forbidden_seen: Set[str] = set()
    unexpected_keys: Set[str] = set()
    records = 0
    if not path.exists():
        return {
            "path_exists": False,
            "records_checked": 0,
            "scan_limit": limit,
            "truncated": False,
            "stage_b_evidence_keys": [],
            "unexpected_stage_b_evidence_keys": [],
            "forbidden_terms_seen": [],
        }
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and records >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records += 1
            evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
            stage_b_keys.update(str(key) for key in evidence)
            unexpected_keys.update(str(key) for key in evidence if str(key) not in TEMPORAL_CODE_STAGE_B_EVIDENCE_ALLOWLIST)
            blob = json.dumps(evidence, sort_keys=True)
            for term in FORBIDDEN_TERMS:
                if term in blob:
                    forbidden_seen.add(term)
    total_records_known = records
    truncated = False
    if limit is not None and records >= limit:
        with path.open("r", encoding="utf-8") as f:
            total_records_known = sum(1 for line in f if line.strip())
        truncated = total_records_known > records
    return {
        "path_exists": True,
        "records_checked": records,
        "total_records_known": total_records_known,
        "scan_limit": limit,
        "truncated": truncated,
        "stage_b_evidence_keys": sorted(stage_b_keys),
        "allowed_stage_b_evidence_keys": sorted(TEMPORAL_CODE_STAGE_B_EVIDENCE_ALLOWLIST),
        "unexpected_stage_b_evidence_keys": sorted(unexpected_keys),
        "forbidden_terms_seen": sorted(forbidden_seen),
    }


def _audit_selector_file(path: Path, function_names: Set[str]) -> Dict[str, Any]:
    function_audits = {}
    missing_functions = []
    for function_name in sorted(function_names):
        names = _function_source_names(path, function_name)
        if not names:
            missing_functions.append(function_name)
        forbidden = sorted(term for term in FORBIDDEN_TERMS if term in names)
        function_audits[function_name] = {
            "forbidden_terms_found": forbidden,
            "function_found": bool(names),
        }
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else None,
        "functions": function_audits,
        "missing_functions": missing_functions,
    }


def build(
    policy_path: Path,
    temporal_code_selector_path: Path,
    stage_b_scored_path: Path,
    output_path: Path,
    md_output_path: Path,
    stage_b_scan_limit: int | None = None,
) -> Dict[str, Any]:
    blockers: List[str] = []
    policy_audit = _audit_selector_file(policy_path, SELECTOR_FUNCTIONS)
    temporal_code_audit = _audit_selector_file(temporal_code_selector_path, TEMPORAL_CODE_SELECTOR_FUNCTIONS)
    for label, audit in (("policy", policy_audit), ("temporal_code", temporal_code_audit)):
        for function_name, row in audit["functions"].items():
            forbidden = row["forbidden_terms_found"]
            if forbidden:
                blockers.append(f"{label}_selector_function_uses_forbidden_terms:{function_name}:{','.join(forbidden)}")
        for function_name in audit["missing_functions"]:
            blockers.append(f"{label}_selector_function_missing:{function_name}")

    stage_b_fields = _read_stage_b_fields(stage_b_scored_path, limit=stage_b_scan_limit)
    if stage_b_fields["forbidden_terms_seen"]:
        blockers.append(f"stage_b_evidence_contains_forbidden_terms:{','.join(stage_b_fields['forbidden_terms_seen'])}")
    if stage_b_fields["unexpected_stage_b_evidence_keys"]:
        blockers.append(f"stage_b_evidence_contains_unexpected_keys:{','.join(stage_b_fields['unexpected_stage_b_evidence_keys'])}")
    if stage_b_fields.get("truncated"):
        blockers.append("stage_b_evidence_scan_truncated")

    report = {
        "schema_version": "selector-utility-leakage-audit-v2",
        "status": "selector_utility_leakage_audit_passed" if not blockers else "selector_utility_leakage_audit_failed",
        "source_sha256": {
            str(policy_path): sha256_file(policy_path),
            str(temporal_code_selector_path): sha256_file(temporal_code_selector_path),
            str(stage_b_scored_path): sha256_file(stage_b_scored_path) if stage_b_scored_path.exists() else None,
        },
        "selector_files": {
            "policy_subsets": policy_audit,
            "temporal_code_selector": temporal_code_audit,
        },
        "stage_b_evidence_scan": stage_b_fields,
        "blockers": blockers,
        "interpretation": (
            "predictive_utility_proxy may exist as a diagnostic Core scorer output, "
            "but it must not appear in policy selectors, temporal-code selectors, "
            "Stage-B objective components, selection scoring, pass/fail decisions, "
            "or Stage-B evidence artifacts."
        ),
        "utility_scope": "Stage C validation only; never selector objective",
    }
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Selector Utility Leakage Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        report["interpretation"],
        "",
        "## Selector Functions",
        "",
        "| Selector File | Function | Found | Forbidden Terms Found |",
        "| --- | --- | --- | --- |",
    ]
    for selector_name, audit in report["selector_files"].items():
        for name, row in audit["functions"].items():
            terms = ", ".join(row["forbidden_terms_found"]) or "None"
            found = "yes" if row["function_found"] else "no"
            lines.append(f"| `{selector_name}` | `{name}` | `{found}` | `{terms}` |")
    lines.extend(["", "## Stage-B Evidence", ""])
    sample = report["stage_b_evidence_scan"]
    lines.append(f"- Records checked: `{sample['records_checked']}`")
    lines.append(f"- Total records known: `{sample.get('total_records_known')}`")
    lines.append(f"- Truncated: `{sample['truncated']}`")
    lines.append(f"- Unexpected keys: `{', '.join(sample['unexpected_stage_b_evidence_keys']) or 'None'}`")
    lines.append(f"- Forbidden terms seen: `{', '.join(sample['forbidden_terms_seen']) or 'None'}`")
    lines.extend(["", "## Blockers", ""])
    lines.extend([f"- `{b}`" for b in report["blockers"]] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build selector Utility leakage audit.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--temporal-code-selector", type=Path, default=DEFAULT_TEMPORAL_CODE_SELECTOR)
    parser.add_argument("--stage-b-scored", type=Path, default=DEFAULT_STAGE_B_SCORED)
    parser.add_argument("--stage-b-scan-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    args = parser.parse_args()
    scan_limit = None if int(args.stage_b_scan_limit) <= 0 else int(args.stage_b_scan_limit)
    report = build(args.policy, args.temporal_code_selector, args.stage_b_scored, args.output, args.md_output, scan_limit)
    print({"status": report["status"], "blockers": report["blockers"]})
    return 0 if not report["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
