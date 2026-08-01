#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file


JsonMap = dict[str, Any]
CONFIG_PATH = Path("configs") / "math_domain_selector_v2_materialization.json"
MATH_TERMS = (
    "$", "\\frac", "\\sum", "\\int", "\\mathbb", " theorem", " proof", " solve",
    " equation", " function", " derivative", " integral", " probability", " sequence", "=",
)
STYLE_PRIOR = {
    "problem_solution": 0.28, "qa_math": 0.26, "math_web_text": 0.18,
    "proof_or_theorem": 0.12, "bibliographic_review": 0.03,
}


def _jsonl(path: Path) -> list[JsonMap]:
    rows: list[JsonMap] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _token(row: JsonMap) -> int:
    value = row.get("token_proxy_count", row.get("token_proxy", 0))
    if isinstance(value, bool):
        return 0
    return int(value) if isinstance(value, int | float | str) and str(value).strip() else 0


def _text(row: JsonMap) -> str:
    return str(row.get("text", ""))


def _math_signal(text: str) -> int:
    lowered = f" {text.lower()} "
    return sum(1 for term in MATH_TERMS if term in lowered)


def _style_bucket(text: str, pool: str) -> str:
    lowered = text.lower()
    if pool == "known_high_quality_reference_pool":
        return "problem_solution"
    if "reviewer:" in lowered or "zbl" in lowered:
        return "bibliographic_review"
    if "?" in text and "$" in text:
        return "qa_math"
    if any(term in lowered for term in ("solve", "find ", "calculate", "answer")) and "$" in text:
        return "qa_math"
    if "proof" in lowered or "theorem" in lowered:
        return "proof_or_theorem"
    return "math_web_text"


def _stable(rows: list[JsonMap], seed: int, label: str) -> list[JsonMap]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{label}:{row['chunk_uid']}".encode()).hexdigest())


def _take_until(rows: list[JsonMap], budget: int) -> list[JsonMap]:
    selected: list[JsonMap] = []
    total = 0
    for row in rows:
        selected.append(row)
        total += _token(row)
        if total >= budget:
            break
    return selected


def _stage0(rows: list[JsonMap], config: JsonMap) -> tuple[list[JsonMap], list[JsonMap]]:
    terms = [str(term).lower() for term in config["stage0"]["blocked_benchmark_terms"]]
    minimum = int(config["stage0"]["minimum_token_proxy"])
    retained: list[JsonMap] = []
    quarantined: list[JsonMap] = []
    for row in rows:
        text = _text(row)
        blockers = []
        if _token(row) < minimum:
            blockers.append("below_minimum_token_proxy")
        if any(term in text.lower() for term in terms):
            blockers.append("benchmark_contamination_term")
        out = {**row, "chunk_uid": f"{row['record_uid']}::chunk-0000", "stage0_blockers": blockers}
        retained.append(out) if not blockers else quarantined.append(out)
    return retained, quarantined


def _stage_a(rows: list[JsonMap], config: JsonMap) -> tuple[list[JsonMap], list[JsonMap]]:
    minimum_tokens = int(config["stage_a"]["minimum_token_proxy"])
    minimum_signal = int(config["stage_a"]["minimum_math_signal_count"])
    passed: list[JsonMap] = []
    rejected: list[JsonMap] = []
    for row in rows:
        text = _text(row)
        signal = _math_signal(text)
        blockers = []
        if _token(row) < minimum_tokens:
            blockers.append("below_stage_a_token_proxy")
        if signal < minimum_signal:
            blockers.append("insufficient_math_signal")
        enriched = {
            **row,
            "chunk_uid": row.get("chunk_uid", f"{row['record_uid']}::chunk-0000"),
            "stage_a_pass": not blockers,
            "stage_a_blockers": blockers,
            "math_signal_count": signal,
            "style_bucket": _style_bucket(text, str(row.get("pool", ""))),
        }
        passed.append(enriched) if not blockers else rejected.append(enriched)
    return passed, rejected


def _score_rows(rows: list[JsonMap], config: JsonMap, budget: int) -> list[JsonMap]:
    seen: set[str] = set()
    scored: list[JsonMap] = []
    max_tokens = min(
        int(config["stage_b"]["max_record_token_proxy"]),
        max(1, int(budget * float(config["stage_b"]["max_single_record_token_share_of_budget"]))),
    )
    for row in rows:
        text = _text(row)
        fingerprint = hashlib.sha256(text[:320].lower().encode("utf-8", errors="replace")).hexdigest()
        duplicate_risk = 1.0 if fingerprint in seen else 0.0
        seen.add(fingerprint)
        length_fit = 1.0 if 80 <= _token(row) <= 1200 else 0.75 if _token(row) <= max_tokens else 0.25
        signal = min(float(row["math_signal_count"]) / 8.0, 1.0)
        style_prior = STYLE_PRIOR.get(str(row["style_bucket"]), 0.0)
        long_doc_penalty = 0.2 if _token(row) > max_tokens else 0.0
        objective = round((0.45 * signal) + (0.25 * length_fit) + style_prior - (0.25 * duplicate_risk) - long_doc_penalty, 6)
        scored.append({**row, "stage_b_evidence": {"objective_score": objective, "duplicate_risk": duplicate_risk}})
    return sorted(scored, key=lambda row: (-float(row["stage_b_evidence"]["objective_score"]), _token(row), row["chunk_uid"]))


def _can_add(row: JsonMap, selected: list[JsonMap], budget: int, caps: JsonMap) -> bool:
    style = str(row["style_bucket"])
    after_style_tokens = sum(_token(item) for item in selected if item.get("style_bucket") == style) + _token(row)
    cap = caps.get(style)
    return cap is None or after_style_tokens <= int(budget * float(cap))


def _stage_b_v2(rows: list[JsonMap], config: JsonMap) -> list[JsonMap]:
    budget = int(sum(_token(row) for row in rows) * float(config["stage_b"]["budget_fraction_of_stage_a_tokens"]))
    scored = _score_rows(rows, config, budget)
    selected: list[JsonMap] = []
    selected_ids: set[str] = set()
    floors = config["stage_b"]["style_token_floors"]
    caps = config["stage_b"]["style_token_caps"]
    for style, fraction in floors.items():
        floor_tokens = int(budget * float(fraction))
        style_rows = [row for row in scored if row["style_bucket"] == style]
        for row in style_rows:
            if str(row["chunk_uid"]) in selected_ids:
                continue
            selected.append({**row, "stage_b_selection_reason": f"style_floor:{style}"})
            selected_ids.add(str(row["chunk_uid"]))
            if sum(_token(item) for item in selected if item.get("style_bucket") == style) >= floor_tokens:
                break
    for row in scored:
        if sum(_token(item) for item in selected) >= budget:
            break
        if str(row["chunk_uid"]) in selected_ids or not _can_add(row, selected, budget, caps):
            continue
        selected.append({**row, "stage_b_selection_reason": "global_fill_under_caps"})
        selected_ids.add(str(row["chunk_uid"]))
    return selected


def _arm(row: JsonMap, arm: str, source: str) -> JsonMap:
    return {
        "arm": arm,
        "chunk_uid": row["chunk_uid"],
        "text": row["text"],
        "token_proxy_count": _token(row),
        "source_pool": source,
        "domain": "math",
        "stage_a_pass": row.get("stage_a_pass"),
        "style_bucket": row.get("style_bucket"),
        "stage_b_evidence": row.get("stage_b_evidence"),
        "stage_b_selection_reason": row.get("stage_b_selection_reason"),
    }


def _summary(rows: list[JsonMap]) -> JsonMap:
    buckets = sorted({str(row.get("style_bucket", "missing")) for row in rows})
    return {
        "records": len(rows),
        "token_proxy_count": sum(_token(row) for row in rows),
        "style_bucket_counts": {bucket: sum(1 for row in rows if row.get("style_bucket") == bucket) for bucket in buckets},
        "style_token_counts": {bucket: sum(_token(row) for row in rows if row.get("style_bucket") == bucket) for bucket in buckets},
    }


def build() -> JsonMap:
    config = load_json(CONFIG_PATH)
    seed = int(config["seed"])
    output_dir = Path(str(config["output_dir"]))
    raw_rows = _jsonl(Path(str(config["input_pools"]["raw_mixed_pool"])))
    reference_rows = _jsonl(Path(str(config["input_pools"]["known_high_quality_reference_pool"])))
    stage0_retained, stage0_quarantined = _stage0(raw_rows, config)
    stage_a_pass, stage_a_rejected = _stage_a(stage0_retained, config)
    reference_pass, _reference_rejected = _stage_a(reference_rows, config)
    curated = _stage_b_v2(stage_a_pass, config)
    curated_ids = {str(row["chunk_uid"]) for row in curated}
    baseline_pool = [row for row in stage_a_pass if str(row["chunk_uid"]) not in curated_ids]
    cap = min(sum(_token(row) for row in curated), sum(_token(row) for row in baseline_pool), sum(_token(row) for row in reference_pass))
    arms = {
        "raw_random_equal_budget": [_arm(row, "raw_random_equal_budget", "raw_mixed_pool") for row in _take_until(_stable(stage0_retained, seed, "raw"), cap)],
        "stageA_random_equal_budget": [_arm(row, "stageA_random_equal_budget", "stageA_pass_disjoint") for row in _take_until(_stable(baseline_pool, seed, "stageA"), cap)],
        "curated_math_v2_equal_budget": [_arm(row, "curated_math_v2_equal_budget", "stage_b_v2_selected") for row in _take_until(curated, cap)],
        "known_high_quality_equal_budget": [_arm(row, "known_high_quality_equal_budget", "reference") for row in _take_until(_stable(reference_pass, seed, "reference"), cap)],
        "raw_full_natural": [_arm(row, "raw_full_natural", "raw_mixed_pool") for row in stage0_retained],
        "stageA_full_natural": [_arm(row, "stageA_full_natural", "stageA_pass_full") for row in stage_a_pass],
        "curated_math_v2_natural": [_arm(row, "curated_math_v2_natural", "stage_b_v2_selected_full") for row in curated],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in arms.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    _write_jsonl(output_dir / "stage0_retained.jsonl", stage0_retained)
    _write_jsonl(output_dir / "stage0_quarantined.jsonl", stage0_quarantined)
    _write_jsonl(output_dir / "stage_a_pass.jsonl", stage_a_pass)
    _write_jsonl(output_dir / "stage_a_rejected.jsonl", stage_a_rejected)
    _write_jsonl(output_dir / "stage_b_v2_selected.jsonl", curated)
    report = {
        "schema_version": "math-domain-selector-v2-materialization-report",
        "status": "math_selector_v2_materialized",
        "training_token_budget_cap": cap,
        "stage_counts": {
            "raw_records": len(raw_rows),
            "stage0_retained": len(stage0_retained),
            "stage0_quarantined": len(stage0_quarantined),
            "stage_a_pass": len(stage_a_pass),
            "stage_a_rejected": len(stage_a_rejected),
            "stage_b_v2_selected": len(curated),
            "stage_a_random_disjoint_pool": len(baseline_pool),
            "reference_stage_a_pass": len(reference_pass),
        },
        "arms": {name: _summary(rows) for name, rows in arms.items()},
        "disjointness": {"curated_stageA_random_disjoint": curated_ids.isdisjoint({str(row["chunk_uid"]) for row in arms["stageA_random_equal_budget"]})},
        "stage_b_distribution_contract": {
            "floors": config["stage_b"]["style_token_floors"],
            "caps": config["stage_b"]["style_token_caps"],
            "max_record_token_proxy": config["stage_b"]["max_record_token_proxy"],
        },
        "selector_forbidden_signals": config["stage_b"]["selector_forbidden_signals"],
        "utility_scope": config["utility_scope"],
        "claim_boundary": config["claim_boundary"],
        "source_sha256": {
            str(CONFIG_PATH): sha256_file(CONFIG_PATH),
            str(config["input_pools"]["raw_mixed_pool"]): sha256_file(Path(str(config["input_pools"]["raw_mixed_pool"]))),
            str(config["input_pools"]["known_high_quality_reference_pool"]): sha256_file(Path(str(config["input_pools"]["known_high_quality_reference_pool"]))),
        },
    }
    save_json(output_dir / "math_selector_v2_materialization_report.json", report)
    save_json(OUTPUT_DIR / "validation" / "math_domain_selector_v2_materialization_report.json", report)
    return report


def main() -> int:
    report = build()
    print(json.dumps({"status": report["status"], "stage_counts": report["stage_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
