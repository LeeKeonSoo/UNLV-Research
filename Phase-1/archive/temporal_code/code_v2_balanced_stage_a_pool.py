#!/usr/bin/env python3
"""Build a deterministic non-Utility balanced Stage-A pool for v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_CONFIG = Path("configs") / "code_domain_next_development_cycle_v2_design.json"
DEFAULT_INPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_combined"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_code_domain_v2_balanced"
SPLITS = ("train", "development", "confirmatory")


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _token_proxy(row: Dict[str, Any]) -> int:
    evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
    return int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or token_proxy_count(str(row.get("text") or "")))


def _stable_key(row: Dict[str, Any], seed: int, label: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{row.get('chunk_uid')}".encode("utf-8")).hexdigest()


def _stratify_test_ratio(rows: List[Dict[str, Any]], target: float, seed: int, split: str) -> List[Dict[str, Any]]:
    tests = [row for row in rows if row.get("content_type") == "test"]
    non_tests = [row for row in rows if row.get("content_type") != "test"]
    tests = sorted(tests, key=lambda row: _stable_key(row, seed, f"{split}:test"))
    non_tests = sorted(non_tests, key=lambda row: _stable_key(row, seed, f"{split}:non_test"))
    if not tests or not non_tests:
        return rows
    current = len(tests) / max(1, len(rows))
    if current < target:
        keep_non_tests = min(len(non_tests), max(1, int(len(tests) * (1.0 - target) / target)))
        selected = tests + non_tests[:keep_non_tests]
    else:
        keep_tests = min(len(tests), max(1, int(len(non_tests) * target / (1.0 - target))))
        selected = non_tests + tests[:keep_tests]
    return sorted(selected, key=lambda row: _stable_key(row, seed, f"{split}:final"))


def _repo_tokens(rows: List[Dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter[str(row.get("repository_identity") or "missing")] += _token_proxy(row)
    return counter


def _apply_repo_token_cap(rows: List[Dict[str, Any]], cap: float, seed: int, split: str) -> List[Dict[str, Any]]:
    selected = list(rows)
    while selected:
        repo_tokens = _repo_tokens(selected)
        total = sum(repo_tokens.values())
        if not total:
            break
        repo, tokens = repo_tokens.most_common(1)[0]
        share = tokens / total
        if share <= cap:
            break
        candidates = [row for row in selected if str(row.get("repository_identity") or "missing") == repo]
        if not candidates:
            break
        # Remove the largest overrepresented chunk first, with deterministic
        # tie-breaking. This preserves smaller examples and reduces share fast.
        remove = max(candidates, key=lambda row: (_token_proxy(row), _stable_key(row, seed, f"{split}:repo_cap")))
        selected.remove(remove)
    return selected


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    repo_tokens = _repo_tokens(rows)
    content_counts = Counter(str(row.get("content_type") or "missing") for row in rows)
    token_sum = sum(_token_proxy(row) for row in rows)
    return {
        "records": len(rows),
        "token_proxy_sum": token_sum,
        "repository_count": len(repo_tokens),
        "largest_repository_token_share": round((repo_tokens.most_common(1)[0][1] / token_sum) if token_sum and repo_tokens else 0.0, 6),
        "content_type_counts": dict(sorted(content_counts.items())),
        "test_record_ratio": round(content_counts.get("test", 0) / max(1, len(rows)), 6),
        "top_repositories": [
            {"repository_identity": repo, "token_proxy": int(tokens), "share": round(tokens / max(1, token_sum), 6)}
            for repo, tokens in repo_tokens.most_common(10)
        ],
    }


def build(config_path: Path, input_dir: Path, output_dir: Path, seed: int, target_test_ratio: float) -> Dict[str, Any]:
    config = load_json(config_path)
    requirements = config["candidate_pool_requirements"]
    cap = float(requirements["maximum_token_share_per_repository"])
    source_sha256 = {
        str(input_dir / split / "stage_a_pass.jsonl"): sha256_file(input_dir / split / "stage_a_pass.jsonl")
        for split in SPLITS
    }
    summaries = {}
    for split in SPLITS:
        rows = list(_jsonl(input_dir / split / "stage_a_pass.jsonl"))
        before = _summarize(rows)
        if split in {"development", "confirmatory"}:
            rows = _stratify_test_ratio(rows, target_test_ratio, seed, split)
        rows = _apply_repo_token_cap(rows, cap, seed, split)
        rows = sorted(rows, key=lambda row: _stable_key(row, seed, f"{split}:output"))
        _write_jsonl(output_dir / split / "stage_a_pass.jsonl", rows)
        _write_jsonl(output_dir / split / "stage_a_rejected.jsonl", [])
        _write_jsonl(output_dir / split / "stage_a_unchunkable.jsonl", [])
        summaries[split] = {"before": before, "after": _summarize(rows)}
    report = {
        "schema_version": "code-domain-v2-balanced-stage-a-pool-v1",
        "status": "balanced_stage_a_pool_built_before_stage_b_or_stage_c",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_sha256": source_sha256,
        "seed": seed,
        "target_development_confirmatory_test_record_ratio": target_test_ratio,
        "maximum_token_share_per_repository": cap,
        "split_summaries": summaries,
        "selection_uses": [
            "Stage-A pass status",
            "repository identity",
            "content_type",
            "token proxy count",
            "deterministic chunk_uid hash"
        ],
        "selection_forbids": [
            "Stage-B outcomes",
            "Utility",
            "benchmark outcomes",
            "retention outcomes",
            "development model outcomes",
            "confirmatory model outcomes",
            "human or LLM review labels"
        ],
        "utility_scope": "Stage C validation only; never selector objective",
        "claim_boundary": "Balanced Stage-A candidate pool only; no Stage-B, Stage-C, Utility, confirmatory, release, or paper success claim.",
    }
    save_json(output_dir / "balanced_stage_a_pool_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain v2 balanced Stage-A pool.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--target-test-ratio", type=float, default=0.30)
    args = parser.parse_args()
    report = build(args.config, args.input_dir, args.output_dir, args.seed, args.target_test_ratio)
    print({"status": report["status"], "split_summaries": report["split_summaries"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
