#!/usr/bin/env python3
"""Freeze code-domain Qwen3-4B development plan and heldout NLL slice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_CONFIG = Path("configs") / "code_domain_development_plan_qwen3_4b_v1.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "code_domain_development_qwen3_4b_v1"
DEFAULT_REPORT = OUTPUT_DIR / "validation" / "code_domain_development_plan_qwen3_4b_report.json"
PROJECT_DIR = Path(__file__).resolve().parents[2]


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_order(records: Iterable[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{row['chunk_uid']}".encode("utf-8")).hexdigest(),
    )


def _freeze_heldout(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    heldout = config["heldout_nll"]
    source = _resolve(heldout["source_file"])
    allowed = set(heldout["allowed_content_types"])
    budget = int(heldout["token_proxy_budget"])
    rows = [
        {**row, "token_proxy_count": token_proxy_count(str(row.get("text") or ""))}
        for row in _jsonl(source)
        if row.get("split") == heldout["source_split"]
        and row.get("stage_a_pass") is True
        and row.get("content_type") in allowed
    ]
    selected = []
    tokens = 0
    for row in _stable_order(rows, int(heldout["seed"])):
        count = int(row["token_proxy_count"])
        if tokens + count > budget and selected:
            continue
        selected.append(row)
        tokens += count
        if tokens >= budget:
            break
    if not selected:
        raise RuntimeError("Development heldout selection produced no records.")
    output = output_dir / "heldouts" / f"{heldout['development_slice_name']}.jsonl"
    _write_jsonl(output, selected)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "candidate_records": len(rows),
        "selected_records": len(selected),
        "selected_token_proxy": tokens,
        "content_type_counts": {
            value: sum(1 for row in selected if row.get("content_type") == value)
            for value in sorted({row.get("content_type") for row in selected})
        },
        "repository_count": len({row.get("repository_identity") for row in selected}),
    }


def freeze(config_path: Path, output_dir: Path, report_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    blockers = []
    inputs = {name: _resolve(path) for name, path in config["inputs"].items()}
    missing = [name for name, path in inputs.items() if not path.exists()]
    blockers.extend(f"missing_input:{name}" for name in missing)
    if blockers:
        report = {
            "schema_version": "code-domain-development-plan-freeze-report-v1",
            "status": "development_plan_blocked",
            "blockers": blockers,
        }
        save_json(report_path, report)
        return report

    arms_report = load_json(inputs["equal_token_arms_report"])
    smoke_report = load_json(inputs["qlora_smoke_report"])
    smoke_blocks = load_json(inputs["qlora_smoke_blocks"])
    evalplus = load_json(inputs["evalplus_guardrail_split"])
    retention = load_json(inputs["retention_guardrails"])
    if arms_report.get("status") != "training_arms_frozen":
        blockers.append("equal_token_arms_not_frozen")
    if smoke_report.get("status") != "qlora_smoke_feasible":
        blockers.append("qlora_smoke_not_feasible")
    if smoke_blocks.get("status") != "frozen_equal_packed_token_blocks":
        blockers.append("qlora_blocks_not_frozen")
    recipe = config["training_recipe"]
    if int(recipe["common_packed_token_budget"]) != int(smoke_blocks["common_packed_token_budget"]):
        blockers.append("packed_token_budget_mismatch")
    if int(recipe["training_token_budget_cap"]) != int(arms_report["summary"]["training_token_budget_cap"]):
        blockers.append("training_token_cap_mismatch")
    if list(recipe["development_training_seeds"]) != list(evalplus["development_training_seeds"]):
        blockers.append("evalplus_seed_contract_mismatch")
    if list(recipe["development_training_seeds"]) != list(retention["seed_contract"]["development_training_seeds"]):
        blockers.append("retention_seed_contract_mismatch")

    heldout_report = _freeze_heldout(config, output_dir)
    source_sha256 = {str(config_path): sha256_file(config_path)}
    source_sha256.update({str(path): sha256_file(path) for path in inputs.values()})
    report = {
        "schema_version": "code-domain-development-plan-freeze-report-v1",
        "status": "development_plan_frozen" if not blockers else "development_plan_blocked",
        "source_sha256": source_sha256,
        "summary": {
            "training_arms": config["training_arms"],
            "primary_comparison": config["primary_comparison"],
            "development_training_seeds": recipe["development_training_seeds"],
            "optimizer_steps": recipe["optimizer_steps"],
            "gradient_accumulation_steps": recipe["gradient_accumulation_steps"],
            "common_packed_token_budget": recipe["common_packed_token_budget"],
            "training_token_budget_cap": recipe["training_token_budget_cap"],
            "heldout": heldout_report,
            "practical_effect_margin": config["practical_effect_margin"],
            "blockers": blockers,
        },
        "external_code_guardrails": config["external_code_guardrails"],
        "general_retention_guardrails": config["general_retention_guardrails"],
        "development_decision_rule": config["development_decision_rule"],
        "forbidden_uses": config["forbidden_uses"],
        "confirmatory_outcomes_read": False,
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
    }
    save_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze code-domain Qwen3-4B development plan.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = freeze(args.config, args.output_dir, args.report)
    print({"status": report["status"], **report.get("summary", {})})
    return 0 if report["status"] == "development_plan_frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
