#!/usr/bin/env python3
"""Build a repository-disjoint silver Redundancy calibration benchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_eval_common import OUTPUT_DIR, save_json
from ingestion.code_chunks import _hard_overlap, token_shingles
from ingestion.code_fingerprints import derived_fingerprints, simhash_hamming_distance
from ingestion.code_selection import token_proxy_count


DEFAULT_STAGE_A = (
    OUTPUT_DIR
    / "temporal_code_collection"
    / "stage_a_code_domain_v2_combined"
    / "train"
    / "stage_a_pass.jsonl"
)
DEFAULT_PAIRS = OUTPUT_DIR / "validation" / "redundancy_real_corpus_silver_pairs.jsonl"
DEFAULT_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_real_corpus_calibration_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "validation" / "redundancy_real_corpus_calibration_report.md"
SIMHASH_THRESHOLDS = [0, 3, 5, 8, 10, 12, 16, 18, 20, 24, 32]
JACCARD_THRESHOLDS = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
CONTAINMENT_THRESHOLDS = [0.70, 0.80, 0.88, 0.90, 0.95]


def _jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            value = line.strip()
            if value:
                row = json.loads(value)
                if isinstance(row, dict):
                    yield row


def _length_bucket(text: str) -> str:
    count = token_proxy_count(text)
    if count < 80:
        return "small"
    if count < 240:
        return "medium"
    return "large"


def _format_only(text: str, is_python: bool) -> str:
    if is_python:
        try:
            return ast.unparse(ast.parse(text)).strip() + "\n"
        except (SyntaxError, ValueError):
            return text
    return " ".join(text.split())


class _SemanticChange(ast.NodeTransformer):
    def __init__(self) -> None:
        self.changed = False

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        if not self.changed and node.ops:
            replacements = {
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
                ast.Lt: ast.GtE,
                ast.LtE: ast.Gt,
                ast.Gt: ast.LtE,
                ast.GtE: ast.Lt,
                ast.In: ast.NotIn,
                ast.NotIn: ast.In,
                ast.Is: ast.IsNot,
                ast.IsNot: ast.Is,
            }
            replacement = replacements.get(type(node.ops[0]))
            if replacement is not None:
                node.ops[0] = replacement()
                self.changed = True
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if not self.changed:
            replacements = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.FloorDiv,
                ast.FloorDiv: ast.Mult,
            }
            replacement = replacements.get(type(node.op))
            if replacement is not None:
                node.op = replacement()
                self.changed = True
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node = self.generic_visit(node)
        if not self.changed:
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
                self.changed = True
            elif isinstance(node.op, ast.Or):
                node.op = ast.And()
                self.changed = True
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not self.changed and isinstance(node.value, bool):
            self.changed = True
            return ast.copy_location(ast.Constant(value=not node.value), node)
        if not self.changed and isinstance(node.value, int) and not isinstance(node.value, bool):
            self.changed = True
            return ast.copy_location(ast.Constant(value=node.value + 1), node)
        return node


def _semantic_change(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    transformer = _SemanticChange()
    changed = transformer.visit(tree)
    if not transformer.changed:
        return None
    ast.fix_missing_locations(changed)
    return ast.unparse(changed).strip() + "\n"


def _containment_extension(text: str, is_python: bool) -> str:
    suffix = (
        "\n# Additional source note preserved with the same implementation unit.\n"
        if is_python
        else " Additional source context records the same behavior for a later release."
    )
    return text.rstrip() + suffix


def _select_sources(rows: List[Dict[str, Any]], per_stratum: int) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        content_type = str(row.get("content_type") or "unknown")
        text = str(row.get("text") or "")
        groups[(content_type, _length_bucket(text))].append(row)
    used_repositories: set[str] = set()
    selected = []
    for stratum in sorted(groups):
        candidates = sorted(groups[stratum], key=lambda row: str(row["chunk_uid"]))
        count = 0
        for row in candidates:
            repository = str(row.get("repository_identity") or "")
            if not repository or repository in used_repositories:
                continue
            selected.append(row)
            used_repositories.add(repository)
            count += 1
            if count >= per_stratum:
                break
    return selected


def _pair(
    pair_id: str,
    label: str,
    transformation: str,
    source: Dict[str, Any],
    left: str,
    right: str,
    *,
    right_repository: str | None = None,
) -> Dict[str, Any]:
    left_fingerprint = derived_fingerprints(left)
    right_fingerprint = derived_fingerprints(right)
    overlap = _hard_overlap(token_shingles(left), token_shingles(right))
    return {
        "pair_id": pair_id,
        "label": label,
        "transformation": transformation,
        "content_type": str(source.get("content_type") or "unknown"),
        "length_bucket": _length_bucket(left),
        "source_chunk_uid": str(source["chunk_uid"]),
        "left_repository": str(source.get("repository_identity") or ""),
        "right_repository": right_repository or str(source.get("repository_identity") or ""),
        "left_token_count": token_proxy_count(left),
        "right_token_count": token_proxy_count(right),
        "left_text": left,
        "right_text": right,
        "exact_match": hashlib.sha256(left.encode("utf-8")).digest()
        == hashlib.sha256(right.encode("utf-8")).digest(),
        "simhash_distance": simhash_hamming_distance(
            str(left_fingerprint["token_simhash64"]),
            str(right_fingerprint["token_simhash64"]),
        ),
        "jaccard": round(float(overlap["jaccard"]), 6),
        "containment": round(float(overlap["containment"]), 6),
    }


def _independent_pair(source: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    source_repo = str(source.get("repository_identity") or "")
    content_type = str(source.get("content_type") or "")
    bucket = _length_bucket(str(source.get("text") or ""))
    candidates = [
        row
        for row in rows
        if str(row.get("repository_identity") or "") != source_repo
        and str(row.get("content_type") or "") == content_type
        and _length_bucket(str(row.get("text") or "")) == bucket
    ]
    if not candidates:
        return None
    source_shingles = token_shingles(str(source.get("text") or ""))
    ranked = sorted(
        candidates,
        key=lambda row: (
            _hard_overlap(source_shingles, token_shingles(str(row.get("text") or "")))["jaccard"],
            str(row["chunk_uid"]),
        ),
    )
    other = ranked[0]
    return _pair(
        f"{source['chunk_uid']}::independent",
        "nonduplicate",
        "cross_repository_independent",
        source,
        str(source["text"]),
        str(other["text"]),
        right_repository=str(other.get("repository_identity") or ""),
    )


def _build_pairs_from_sources(
    rows: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for source in sources:
        text = str(source["text"])
        uid = str(source["chunk_uid"])
        is_python = str(source.get("path") or "").lower().endswith(".py")
        pairs.append(_pair(f"{uid}::exact", "hard_duplicate", "exact_copy", source, text, text))
        formatted = _format_only(text, is_python)
        if formatted != text:
            pairs.append(
                _pair(f"{uid}::format", "hard_duplicate", "format_only", source, text, formatted)
            )
        pairs.append(
            _pair(
                f"{uid}::containment",
                "hard_duplicate",
                "containment_extension",
                source,
                text,
                _containment_extension(text, is_python),
            )
        )
        if is_python:
            changed = _semantic_change(text)
            if changed and changed != text:
                pairs.append(
                    _pair(
                        f"{uid}::semantic-change",
                        "nonduplicate",
                        "semantic_change_control",
                        source,
                        text,
                        changed,
                    )
                )
        independent = _independent_pair(source, rows)
        if independent is not None:
            pairs.append(independent)
    metadata = {
        "source_count": len(sources),
        "source_repository_count": len({str(row.get("repository_identity") or "") for row in sources}),
        "source_strata": dict(
            sorted(
                Counter(
                    f"{row.get('content_type')}::{_length_bucket(str(row.get('text') or ''))}"
                    for row in sources
                ).items()
            )
        ),
        "source_chunk_uids": [str(row["chunk_uid"]) for row in sources],
    }
    return pairs, metadata


def _build_pairs(rows: List[Dict[str, Any]], per_stratum: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    return _build_pairs_from_sources(rows, _select_sources(rows, per_stratum))


def _predict(
    row: Dict[str, Any],
    simhash_threshold: int,
    jaccard_threshold: float,
    containment_threshold: float,
) -> bool:
    return bool(
        row["exact_match"]
        or (
            int(row["simhash_distance"]) <= simhash_threshold
            and (
                float(row["jaccard"]) >= jaccard_threshold
                or float(row["containment"]) >= containment_threshold
            )
        )
    )


def _metrics(rows: List[Dict[str, Any]], threshold: Dict[str, Any]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        expected = row["label"] == "hard_duplicate"
        predicted = _predict(
            row,
            int(threshold["simhash_threshold"]),
            float(threshold["jaccard_threshold"]),
            float(threshold["containment_threshold"]),
        )
        if expected and predicted:
            tp += 1
        elif expected:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "pair_count": len(rows),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "useful_data_dropout_rate": round(fp / max(1, fp + tn), 6),
    }


def _sweep(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for simhash in SIMHASH_THRESHOLDS:
        for jaccard in JACCARD_THRESHOLDS:
            for containment in CONTAINMENT_THRESHOLDS:
                threshold = {
                    "simhash_threshold": simhash,
                    "jaccard_threshold": jaccard,
                    "containment_threshold": containment,
                }
                results.append({**threshold, **_metrics(rows, threshold)})
    return sorted(
        results,
        key=lambda row: (
            -row["f1"],
            row["useful_data_dropout_rate"],
            -row["precision"],
            -row["recall"],
            row["simhash_threshold"],
        ),
    )


def _stratified(rows: List[Dict[str, Any]], threshold: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    dimensions = {
        "content_type": sorted({str(row["content_type"]) for row in rows}),
        "length_bucket": sorted({str(row["length_bucket"]) for row in rows}),
        "transformation": sorted({str(row["transformation"]) for row in rows}),
    }
    for dimension, values in dimensions.items():
        result[dimension] = {
            value: _metrics([row for row in rows if str(row[dimension]) == value], threshold)
            for value in values
        }
    return result


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    stage_a_path: Path,
    pairs_path: Path,
    output_path: Path,
    md_output_path: Path,
    *,
    per_stratum: int,
) -> Dict[str, Any]:
    rows = list(_jsonl(stage_a_path))
    pairs, source_metadata = _build_pairs(rows, per_stratum)
    current_threshold = {
        "simhash_threshold": 3,
        "jaccard_threshold": 0.75,
        "containment_threshold": 0.88,
    }
    sweep = _sweep(pairs)
    current = {**current_threshold, **_metrics(pairs, current_threshold)}
    near_only = [row for row in pairs if row["transformation"] != "exact_copy"]
    report = {
        "schema_version": "redundancy-real-corpus-calibration-report-v1",
        "status": "redundancy_real_corpus_silver_calibration_ready",
        "claim_boundary": (
            "Repository-disjoint source sampling with deterministic metamorphic silver labels. "
            "This calibrates operational behavior but is not human-validated semantic-clone ground truth."
        ),
        "stage_a_source": str(stage_a_path),
        "pairs_path": str(pairs_path),
        "source_metadata": source_metadata,
        "summary": {
            "pair_count": len(pairs),
            "label_counts": dict(sorted(Counter(str(row["label"]) for row in pairs).items())),
            "transformation_counts": dict(
                sorted(Counter(str(row["transformation"]) for row in pairs).items())
            ),
            "current_threshold": current,
            "current_threshold_near_only": _metrics(near_only, current_threshold),
            "best_silver_threshold": sweep[0],
        },
        "current_stratified": _stratified(pairs, current_threshold),
        "best_stratified": _stratified(pairs, sweep[0]),
        "threshold_sweep_top20": sweep[:20],
        "current_false_negatives": [
            {
                "pair_id": row["pair_id"],
                "transformation": row["transformation"],
                "content_type": row["content_type"],
                "length_bucket": row["length_bucket"],
                "simhash_distance": row["simhash_distance"],
                "jaccard": row["jaccard"],
                "containment": row["containment"],
            }
            for row in pairs
            if row["label"] == "hard_duplicate"
            and not _predict(row, 3, 0.75, 0.88)
        ],
        "current_false_positives": [
            {
                "pair_id": row["pair_id"],
                "transformation": row["transformation"],
                "content_type": row["content_type"],
                "length_bucket": row["length_bucket"],
                "simhash_distance": row["simhash_distance"],
                "jaccard": row["jaccard"],
                "containment": row["containment"],
            }
            for row in pairs
            if row["label"] == "nonduplicate"
            and _predict(row, 3, 0.75, 0.88)
        ],
        "decision": (
            "Do not promote a threshold from silver evidence alone. Use this report to freeze "
            "candidate threshold arms and validate them on an independently generated heldout silver set."
        ),
        "required_next_evidence": [
            "freeze_independent_repository_disjoint_silver_holdout",
            "add_real_cross_repository_template_and_boilerplate_clusters",
            "add_semantic_related_but_useful_controls_beyond_single_operator_mutations",
            "measure_cluster_level_representative_dropout",
            "run_threshold_arms_as_stage_a_development_ablations_before_any_promotion",
        ],
    }
    _write_jsonl(pairs_path, pairs)
    save_json(output_path, report)
    md_output_path.parent.mkdir(parents=True, exist_ok=True)
    md_output_path.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    current = summary["current_threshold"]
    near = summary["current_threshold_near_only"]
    best = summary["best_silver_threshold"]
    lines = [
        "# Real-Corpus Redundancy Silver Calibration",
        "",
        f"Status: `{report['status']}`",
        "",
        str(report["claim_boundary"]),
        "",
        "## Sources",
        "",
        f"- Sources: `{report['source_metadata']['source_count']}`",
        f"- Source repositories: `{report['source_metadata']['source_repository_count']}`",
        f"- Strata: `{report['source_metadata']['source_strata']}`",
        "",
        "## Current Threshold",
        "",
        f"- Pairs: `{summary['pair_count']}`",
        f"- Precision / recall / F1: `{current['precision']} / {current['recall']} / {current['f1']}`",
        f"- Useful-data dropout rate: `{current['useful_data_dropout_rate']}`",
        f"- Near-only recall: `{near['recall']}`",
        f"- False negatives: `{current['false_negative']}`",
        f"- False positives: `{current['false_positive']}`",
        "",
        "## Best Silver Threshold",
        "",
        f"- SimHash / Jaccard / containment: `{best['simhash_threshold']} / {best['jaccard_threshold']} / {best['containment_threshold']}`",
        f"- Precision / recall / F1: `{best['precision']} / {best['recall']} / {best['f1']}`",
        f"- Useful-data dropout rate: `{best['useful_data_dropout_rate']}`",
        "",
        "## Current Stratified Results",
        "",
    ]
    for dimension, groups in report["current_stratified"].items():
        lines.append(f"### {dimension}")
        lines.append("")
        lines.append("| Group | Pairs | Precision | Recall | Dropout |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, metrics in groups.items():
            lines.append(
                f"| `{name}` | `{metrics['pair_count']}` | `{metrics['precision']}` | "
                f"`{metrics['recall']}` | `{metrics['useful_data_dropout_rate']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            str(report["decision"]),
            "",
            "## Required Next Evidence",
            "",
        ]
    )
    lines.extend([f"- `{value}`" for value in report["required_next_evidence"]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real-corpus Redundancy silver calibration.")
    parser.add_argument("--stage-a", type=Path, default=DEFAULT_STAGE_A)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--per-stratum", type=int, default=3)
    args = parser.parse_args()
    report = build(
        args.stage_a,
        args.pairs,
        args.output,
        args.md_output,
        per_stratum=max(1, args.per_stratum),
    )
    print({"status": report["status"], "summary": report["summary"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
