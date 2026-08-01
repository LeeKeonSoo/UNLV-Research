#!/usr/bin/env python3
"""Build the code-domain v2 candidate-pool readiness report."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, load_json, save_json, sha256_file
from ingestion.code_selection import token_proxy_count


DEFAULT_CONFIG = Path("configs") / "code_domain_next_development_cycle_v2_design.json"
DEFAULT_STAGE_A_DIR = OUTPUT_DIR / "temporal_code_collection" / "stage_a_path_stratified_tranche"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "code_domain_v2_candidate_pool_readiness_report.json"
DEFAULT_DOC = Path("docs") / "code_domain_v2_candidate_pool_readiness.md"


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                row = json.loads(raw)
                if isinstance(row, dict):
                    yield row


def _token_proxy(row: Dict[str, Any]) -> int:
    evidence = row.get("stage_b_evidence") if isinstance(row.get("stage_b_evidence"), dict) else {}
    return int(evidence.get("token_proxy_count") or row.get("token_proxy_count") or token_proxy_count(str(row.get("text") or "")))


def _ratio(counter: Counter[str], key: str) -> float:
    total = sum(counter.values())
    return float(counter.get(key, 0)) / max(1, total)


def _shares(counter: Counter[str]) -> Dict[str, float]:
    total = sum(counter.values())
    return {key: round(float(value) / max(1, total), 6) for key, value in sorted(counter.items())}


def _top_shares(counter: Counter[str], limit: int = 10) -> List[Dict[str, Any]]:
    total = sum(counter.values())
    return [
        {"value": key, "token_proxy": int(value), "share": round(float(value) / max(1, total), 6)}
        for key, value in counter.most_common(limit)
    ]


def _profile_split(path: Path) -> Dict[str, Any]:
    records = list(_jsonl(path))
    token_counts = [_token_proxy(row) for row in records]
    repo_tokens: Counter[str] = Counter()
    repo_records: Counter[str] = Counter()
    bundle_tokens: Counter[str] = Counter()
    content_records: Counter[str] = Counter()
    content_tokens: Counter[str] = Counter()
    chunk_kind_records: Counter[str] = Counter()
    change_type_records: Counter[str] = Counter()
    suffix_records: Counter[str] = Counter()

    for row, tokens in zip(records, token_counts):
        repo = str(row.get("repository_identity") or "missing")
        bundle = str(row.get("bundle_id") or "missing")
        content_type = str(row.get("content_type") or "missing")
        chunk_kind = str(row.get("chunk_kind") or "missing")
        change_type = str(row.get("change_type") or "missing")
        suffix = Path(str(row.get("path") or "")).suffix or "no_suffix"
        repo_tokens[repo] += tokens
        repo_records[repo] += 1
        bundle_tokens[bundle] += tokens
        content_records[content_type] += 1
        content_tokens[content_type] += tokens
        chunk_kind_records[chunk_kind] += 1
        change_type_records[change_type] += 1
        suffix_records[suffix] += 1

    token_sum = sum(token_counts)
    largest_repo_share = max(_shares(repo_tokens).values(), default=0.0)
    return {
        "path": str(path),
        "sha256": sha256_file(path) if path.exists() else None,
        "records": len(records),
        "token_proxy_sum": token_sum,
        "token_proxy_mean": round(float(token_sum) / max(1, len(records)), 6),
        "token_proxy_median": statistics.median(token_counts) if token_counts else 0,
        "repository_count": len(repo_tokens),
        "repository_record_counts": dict(sorted(repo_records.items())),
        "top_repository_token_shares": _top_shares(repo_tokens),
        "largest_repository_token_share": largest_repo_share,
        "top_bundle_token_shares": _top_shares(bundle_tokens),
        "content_type_counts": dict(sorted(content_records.items())),
        "content_type_record_ratios": _shares(content_records),
        "content_type_token_ratios": _shares(content_tokens),
        "test_record_ratio": round(_ratio(content_records, "test"), 6),
        "test_token_ratio": round(_ratio(content_tokens, "test"), 6),
        "chunk_kind_counts": dict(sorted(chunk_kind_records.items())),
        "chunk_kind_ratios": _shares(chunk_kind_records),
        "change_type_counts": dict(sorted(change_type_records.items())),
        "path_suffix_counts": dict(sorted(suffix_records.items())),
        "repository_set": sorted(repo_tokens),
    }


def _check_repo_disjointness(profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pairs = [
        ("train", "development"),
        ("train", "confirmatory"),
        ("development", "confirmatory"),
    ]
    overlaps: Dict[str, List[str]] = {}
    for left, right in pairs:
        left_set = set(profiles[left]["repository_set"])
        right_set = set(profiles[right]["repository_set"])
        overlaps[f"{left}_vs_{right}"] = sorted(left_set.intersection(right_set))
    return {
        "repository_overlaps": overlaps,
        "repository_disjoint": all(not values for values in overlaps.values()),
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Code-Domain v2 Candidate-Pool Readiness",
        "",
        "## Status",
        "",
        f"- Status: `{report['status']}`.",
        f"- Stage-A source: `{report['inputs']['stage_a_dir']}`.",
        f"- Blockers: `{len(report['blockers'])}`.",
        "",
        "## Split Summary",
        "",
        "| Split | Records | Tokens | Repositories | Largest repo share | Test ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "development", "confirmatory"):
        profile = report["split_profiles"][split]
        lines.append(
            "| {split} | {records} | {tokens} | {repos} | {largest:.6f} | {test:.6f} |".format(
                split=split,
                records=profile["records"],
                tokens=profile["token_proxy_sum"],
                repos=profile["repository_count"],
                largest=profile["largest_repository_token_share"],
                test=profile["test_record_ratio"],
            )
        )
    lines.extend(["", "## Blockers", ""])
    if report["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in report["blockers"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "Utility remains Stage C validation only and never a selector objective.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(config_path: Path, stage_a_dir: Path, output_path: Path, doc_path: Path) -> Dict[str, Any]:
    config = load_json(config_path)
    requirements = config["candidate_pool_requirements"]
    min_repos = requirements["minimum_stage_a_pass_repositories"]
    max_repo_share = float(requirements["maximum_token_share_per_repository"])
    min_holdout_tokens = int(requirements["heldout_token_proxy_budget"]["minimum_each_split"])
    preferred_holdout_tokens = int(requirements["heldout_token_proxy_budget"]["preferred_each_split"])
    max_test_ratio_diff = float(requirements["maximum_development_confirmatory_test_ratio_difference"])

    split_paths = {
        split: stage_a_dir / split / "stage_a_pass.jsonl"
        for split in ("train", "development", "confirmatory")
    }
    missing_inputs = [str(path) for path in split_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing Stage-A pass inputs: {missing_inputs}")

    profiles = {split: _profile_split(path) for split, path in split_paths.items()}
    disjointness = _check_repo_disjointness(profiles)
    blockers: List[str] = []
    warnings: List[str] = []

    for split in ("train", "development", "confirmatory"):
        observed = int(profiles[split]["repository_count"])
        required_key = "train" if split == "train" else f"{split}_heldout"
        required = int(min_repos[required_key])
        if observed < required:
            blockers.append(f"insufficient_stage_a_pass_repositories:{split}:{observed}<required:{required}")
        largest_share = float(profiles[split]["largest_repository_token_share"])
        if largest_share > max_repo_share:
            blockers.append(f"repository_token_share_cap_exceeded:{split}:{largest_share:.6f}>cap:{max_repo_share}")

    for split in ("development", "confirmatory"):
        tokens = int(profiles[split]["token_proxy_sum"])
        if tokens < min_holdout_tokens:
            blockers.append(f"heldout_token_proxy_below_minimum:{split}:{tokens}<required:{min_holdout_tokens}")
        elif tokens < preferred_holdout_tokens:
            warnings.append(f"heldout_token_proxy_below_preferred:{split}:{tokens}<preferred:{preferred_holdout_tokens}")

    if not disjointness["repository_disjoint"]:
        blockers.append("repository_split_overlap_detected")

    test_ratio_diff = abs(float(profiles["development"]["test_record_ratio"]) - float(profiles["confirmatory"]["test_record_ratio"]))
    if test_ratio_diff > max_test_ratio_diff:
        blockers.append(f"development_confirmatory_test_ratio_diff_exceeds_limit:{test_ratio_diff:.6f}>limit:{max_test_ratio_diff}")

    base_nll_status = {
        "status": "required_later_before_development_promotion",
        "reason": "Base-NLL scale is a Stage-C/development diagnostic and is not available from Stage-A candidate-pool files alone.",
        "selector_use": "forbidden",
    }

    status = "candidate_pool_ready_for_v2_development_design" if not blockers else "candidate_pool_not_ready_for_v2_development_design"
    report = {
        "schema_version": "code-domain-v2-candidate-pool-readiness-v1",
        "status": status,
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "stage_a_dir": str(stage_a_dir),
            "split_paths": {split: str(path) for split, path in split_paths.items()},
        },
        "locked_prior_result": {
            "source_postmortem": config["source_postmortem"]["path"],
            "v1_status": config["source_postmortem"]["locked_v1_status"],
            "v1_interpretation": config["source_postmortem"]["locked_v1_interpretation"],
            "v1_result_can_only_inform_separate_cycle": True,
        },
        "requirements": {
            "split_contract": requirements["split_contract"],
            "minimum_stage_a_pass_repositories": min_repos,
            "maximum_token_share_per_repository": max_repo_share,
            "heldout_token_proxy_budget": requirements["heldout_token_proxy_budget"],
            "maximum_development_confirmatory_test_ratio_difference": max_test_ratio_diff,
            "insufficient_data_action": requirements["insufficient_data_action"],
        },
        "requirement_checks": {
            "repository_disjointness": disjointness,
            "development_confirmatory_test_record_ratio_difference": round(test_ratio_diff, 6),
            "base_nll_scale": base_nll_status,
        },
        "split_profiles": profiles,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": (
            [
                "Proceed to freeze v2 Stage-B arms and development heldouts before any v2 development training outcomes.",
                "Run base-NLL scale diagnostics before v2 promotion decisions.",
                "Keep Utility, benchmark, retention, and model outcomes out of Stage B.",
            ]
            if not blockers
            else [
                "Do not spend more GPU training on this pool as a v2 confirmatory candidate.",
                "Expand or rebuild the raw-like Python corpus until repository and stratification blockers clear.",
                "Re-run Stage 0, Stage A, and this readiness report after corpus expansion.",
                "Only then freeze Stage-B v2 arms and development heldouts.",
            ]
        ),
        "interpretation": (
            "The current Stage-A candidate pool satisfies the v2 corpus-shape requirements."
            if not blockers
            else "The current Stage-A candidate pool is useful diagnostic evidence, but it is not yet shaped well enough for the v2 development cycle."
        ),
        "stage_boundaries": config["stage_boundaries"],
        "selector_signal_policy": config["selector_signal_policy"],
        "confirmatory_outcomes_read_for_v2": config["confirmatory_outcomes_read_for_v2"],
        "utility_scope": config["stage_boundaries"]["utility_scope"],
        "claim_boundary": "Candidate-pool readiness only. No Stage-B, Stage-C, Utility, confirmatory, release, or paper success claim.",
    }
    save_json(output_path, report)
    _write_markdown(doc_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build code-domain v2 candidate-pool readiness report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-a-dir", type=Path, default=DEFAULT_STAGE_A_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    report = build(args.config, args.stage_a_dir, args.output, args.doc)
    print(
        {
            "status": report["status"],
            "blockers": report["blockers"],
            "warnings": report["warnings"],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
