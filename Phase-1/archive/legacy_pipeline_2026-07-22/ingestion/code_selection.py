"""Core-only Stage-B scoring and selection for temporal-code chunks."""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List

from ingestion.code_chunks import token_shingles
from policy.dispositions import (
    BUDGET_NOT_SELECTED,
    annotate_retained_pool,
    disposition_summary,
)


TOKEN_RE = re.compile(r"\w+|[^\s\w]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?")


def token_proxy_count(text: str) -> int:
    return max(1, len(TOKEN_RE.findall(text)))


def _band_support(count: int, low: int, ideal_low: int, ideal_high: int, high: int) -> float:
    if count <= low:
        return 0.0
    if count < ideal_low:
        return (count - low) / max(1, ideal_low - low)
    if count <= ideal_high:
        return 1.0
    if count < high:
        return 1.0 - (0.35 * ((count - ideal_high) / max(1, high - ideal_high)))
    return 0.65


def path_family(path_value: str) -> str:
    parts = PurePosixPath(str(path_value).replace("\\", "/")).parts
    if not parts:
        return "unknown"
    if parts[0].lower() in {"test", "tests"}:
        return parts[0]
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def difficulty_band(record: Dict[str, Any], token_count: int | None = None) -> str:
    count = int(token_count or token_proxy_count(str(record.get("text") or "")))
    if count < 80:
        return "small"
    if count < 240:
        return "medium"
    return "large"


def _python_quality(text: str, token_count: int) -> Dict[str, float]:
    tree = ast.parse(text)
    nodes = list(ast.walk(tree))
    semantic_token_count = token_proxy_count(ast.unparse(tree))
    node_types = {type(node).__name__ for node in nodes}
    names = [node.id for node in nodes if isinstance(node, ast.Name)]
    identifiers = len(set(names)) / max(1, len(names))
    structural_richness = min(1.0, len(node_types) / 14.0)
    length_support = _band_support(semantic_token_count, 8, 28, 320, 900)
    assignments = [node for node in nodes if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
    pass_through_assignments = [
        node
        for node in assignments
        if isinstance(getattr(node, "value", None), ast.Name)
    ]
    pass_through_ratio = len(pass_through_assignments) / max(1, len(assignments))
    quality = (
        0.45 * length_support
        + 0.35 * structural_richness
        + 0.20 * identifiers
        - 0.25 * pass_through_ratio
    )
    return {
        "length_support": length_support,
        "structural_richness": structural_richness,
        "lexical_or_identifier_diversity": identifiers,
        "code_quality_proxy": max(0.0, quality),
        "ast_node_count": float(len(nodes)),
        "semantic_token_proxy_count": float(semantic_token_count),
        "pass_through_assignment_ratio": pass_through_ratio,
    }


def _documentation_quality(text: str, token_count: int) -> Dict[str, float]:
    words = [value.lower() for value in WORD_RE.findall(text)]
    diversity = len(set(words)) / max(1, len(words))
    technical_markers = len(re.findall(r"``|:class:|:func:|:mod:|code-block::|[A-Za-z_]\w*\(", text))
    structural_richness = min(1.0, technical_markers / 6.0)
    length_support = _band_support(token_count, 20, 60, 420, 1000)
    quality = 0.40 * length_support + 0.20 * diversity + 0.40 * structural_richness
    return {
        "length_support": length_support,
        "structural_richness": structural_richness,
        "lexical_or_identifier_diversity": diversity,
        "code_quality_proxy": quality,
        "ast_node_count": 0.0,
        "semantic_token_proxy_count": float(token_count),
        "pass_through_assignment_ratio": 0.0,
    }


def local_stage_b_features(record: Dict[str, Any]) -> Dict[str, Any]:
    text = str(record.get("text") or "")
    count = token_proxy_count(text)
    is_python = str(record.get("path") or "").lower().endswith(".py")
    quality = _python_quality(text, count) if is_python else _documentation_quality(text, count)
    return {
        **{key: round(float(value), 6) for key, value in quality.items()},
        "token_proxy_count": count,
        "coverage_buckets": {
            "bundle_id": str(record.get("bundle_id") or "unknown"),
            "content_type": str(record.get("content_type") or "unknown"),
            "change_type": str(record.get("change_type") or "unknown"),
            "path_family": path_family(str(record.get("path") or "")),
            "difficulty_band": difficulty_band(record, count),
        },
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left.intersection(right)) / len(left.union(right))


class _StageBStructuralNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_name", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_arg", annotation=self.visit(node.annotation) if node.annotation else None), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        value = self.generic_visit(node)
        value.name = "_function"
        return value

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        value = self.generic_visit(node)
        value.name = "_function"
        return value

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        value = self.generic_visit(node)
        value.name = "_class"
        return value

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            value: Any = "_str"
        elif isinstance(node.value, bytes):
            value = b"_bytes"
        elif isinstance(node.value, (int, float, complex)):
            value = 0
        else:
            value = node.value
        return ast.copy_location(ast.Constant(value=value), node)


def _python_structural_shape(record: Dict[str, Any]) -> str | None:
    if not str(record.get("path") or "").lower().endswith(".py"):
        return None
    try:
        tree = ast.parse(str(record.get("text") or ""))
    except SyntaxError:
        return None
    normalized = _StageBStructuralNormalizer().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(normalized)
    return hashlib.sha256(
        ast.dump(normalized, annotate_fields=True, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _structural_saturation_risk(match_count: int, mode: str) -> float:
    count = max(0, int(match_count))
    if count == 0:
        return 0.0
    if mode == "binary_current":
        return 0.85
    if mode == "exp_tau_1":
        return 1.0 - math.exp(-count)
    if mode == "exp_tau_2":
        return 1.0 - math.exp(-count / 2.0)
    if mode == "log_count":
        return min(1.0, 0.55 + 0.20 * math.log2(1 + count))
    raise ValueError(f"Unsupported structural saturation mode: {mode}")


def _redundancy_evidence(
    shingles: List[set[str]],
    structural_shapes: List[str | None],
    mode: str,
    structural_saturation_mode: str = "binary_current",
) -> List[Dict[str, Any]]:
    if mode not in {"all_pairs_exact", "indexed_exact"}:
        raise ValueError(f"Unsupported redundancy search mode: {mode}")
    shape_counts = Counter(shape for shape in structural_shapes if shape is not None)
    inverted: Dict[str, set[int]] = defaultdict(set)
    if mode == "indexed_exact":
        for index, values in enumerate(shingles):
            for value in values:
                inverted[value].add(index)
    results = []
    for index in range(len(shingles)):
        if mode == "all_pairs_exact":
            candidates = [other for other in range(len(shingles)) if other != index]
        else:
            candidate_set: set[int] = set()
            for value in shingles[index]:
                candidate_set.update(inverted[value])
            candidate_set.discard(index)
            candidates = sorted(candidate_set)
        lexical_overlap = max(
            (_overlap(shingles[index], shingles[other]) for other in candidates),
            default=0.0,
        )
        shape = structural_shapes[index]
        structural_match_count = max(0, shape_counts.get(shape, 0) - 1) if shape is not None else 0
        structural_risk = _structural_saturation_risk(
            structural_match_count,
            structural_saturation_mode,
        )
        results.append(
            {
                "soft_lexical_redundancy_risk": lexical_overlap,
                "soft_structural_redundancy_risk": structural_risk,
                "soft_structural_match_count": structural_match_count,
                "soft_redundancy_risk": max(lexical_overlap, structural_risk),
                "structural_saturation_mode": structural_saturation_mode,
                "redundancy_search_mode": mode,
                "lexical_candidate_count": len(candidates),
            }
        )
    return results


def score_stage_b(
    records: List[Dict[str, Any]],
    quality_weight: float,
    redundancy_weight: float,
    redundancy_search_mode: str = "indexed_exact",
    structural_saturation_mode: str = "binary_current",
) -> List[Dict[str, Any]]:
    local = [local_stage_b_features(record) for record in records]
    shingles = [token_shingles(str(record.get("text") or "")) for record in records]
    structural_shapes = [_python_structural_shape(record) for record in records]
    redundancy = _redundancy_evidence(
        shingles,
        structural_shapes,
        redundancy_search_mode,
        structural_saturation_mode,
    )
    scored = []
    for record, features, redundancy_row in zip(records, local, redundancy):
        soft_risk = float(redundancy_row["soft_redundancy_risk"])
        soft_support = 1.0 - soft_risk
        objective = quality_weight * features["code_quality_proxy"] + redundancy_weight * soft_support
        scored.append(
            {
                **record,
                "stage_b_evidence": {
                    **features,
                    "soft_lexical_redundancy_risk": round(float(redundancy_row["soft_lexical_redundancy_risk"]), 6),
                    "soft_structural_redundancy_risk": round(float(redundancy_row["soft_structural_redundancy_risk"]), 6),
                    "soft_structural_match_count": int(redundancy_row["soft_structural_match_count"]),
                    "soft_redundancy_risk": round(soft_risk, 6),
                    "soft_redundancy_support": round(soft_support, 6),
                    "structural_saturation_mode": redundancy_row["structural_saturation_mode"],
                    "redundancy_search_mode": redundancy_row["redundancy_search_mode"],
                    "lexical_candidate_count": int(redundancy_row["lexical_candidate_count"]),
                    "stage_b_objective_score": round(objective, 6),
                },
            }
        )
    return scored


def _stable_order(records: Iterable[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: hashlib.sha256(f"{seed}:{row['chunk_uid']}".encode("utf-8")).hexdigest(),
    )


def _coverage_counts(records: Iterable[Dict[str, Any]], axes: List[str]) -> Dict[str, Dict[str, int]]:
    return {
        axis: dict(
            sorted(
                Counter(
                    str((row.get("stage_b_evidence") or {}).get("coverage_buckets", {}).get(axis) or "unknown")
                    for row in records
                ).items()
            )
        )
        for axis in axes
    }


def _coverage_token_counts(records: Iterable[Dict[str, Any]], axes: List[str]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    rows = list(records)
    for axis in axes:
        counts: Counter[str] = Counter()
        for row in rows:
            bucket = str((row.get("stage_b_evidence") or {}).get("coverage_buckets", {}).get(axis) or "unknown")
            counts[bucket] += int((row.get("stage_b_evidence") or {}).get("token_proxy_count") or 0)
        result[axis] = dict(sorted(counts.items()))
    return result


def select_stage_b(
    records: List[Dict[str, Any]],
    *,
    budget_fraction: float | None,
    quality_weight: float,
    redundancy_weight: float,
    coverage_axes: List[str],
    minimum_exemplars: int,
    baseline_seed: int,
    distribution_axes: List[str] | None = None,
    minimum_relative_token_share: float = 0.0,
    redundancy_search_mode: str = "indexed_exact",
    structural_saturation_mode: str = "binary_current",
) -> Dict[str, Any]:
    scored = score_stage_b(
        records,
        quality_weight,
        redundancy_weight,
        redundancy_search_mode,
        structural_saturation_mode,
    )
    total_tokens = sum(row["stage_b_evidence"]["token_proxy_count"] for row in scored)
    budget_applied = budget_fraction is not None and float(budget_fraction) < 1.0
    effective_fraction = 1.0 if budget_fraction is None else min(1.0, max(0.0, float(budget_fraction)))
    budget = total_tokens if not budget_applied else max(1, math.floor(total_tokens * effective_fraction))
    ranked = sorted(scored, key=lambda row: (-row["stage_b_evidence"]["stage_b_objective_score"], row["chunk_uid"]))
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_tokens = 0

    def add(row: Dict[str, Any], accepted_by: str) -> bool:
        nonlocal selected_tokens
        uid = str(row["chunk_uid"])
        count = int(row["stage_b_evidence"]["token_proxy_count"])
        if uid in selected_ids or selected_tokens + count > budget:
            return False
        selected_ids.add(uid)
        selected_tokens += count
        selected.append({**row, "stage_b_selection": {"selected": True, "accepted_by": accepted_by}})
        return True

    for axis in coverage_axes:
        values = sorted(
            {
                str(row["stage_b_evidence"]["coverage_buckets"].get(axis) or "unknown")
                for row in ranked
            }
        )
        for value in values:
            candidates = [
                row
                for row in ranked
                if str(row["stage_b_evidence"]["coverage_buckets"].get(axis) or "unknown") == value
            ]
            retained = sum(
                str(row["stage_b_evidence"]["coverage_buckets"].get(axis) or "unknown") == value
                for row in selected
            )
            for row in candidates:
                if retained >= max(1, minimum_exemplars):
                    break
                if add(row, f"coverage:{axis}:{value}"):
                    retained += 1
    source_token_counts = _coverage_token_counts(scored, distribution_axes or [])
    total_source_tokens = sum(row["stage_b_evidence"]["token_proxy_count"] for row in scored)
    for axis in distribution_axes or []:
        for value, source_tokens in source_token_counts[axis].items():
            required = math.ceil((source_tokens / max(1, total_source_tokens)) * budget * minimum_relative_token_share)
            retained_tokens = sum(
                int(row["stage_b_evidence"]["token_proxy_count"])
                for row in selected
                if str(row["stage_b_evidence"]["coverage_buckets"].get(axis) or "unknown") == value
            )
            for row in ranked:
                if retained_tokens >= required:
                    break
                if str(row["stage_b_evidence"]["coverage_buckets"].get(axis) or "unknown") != value:
                    continue
                if add(row, f"distribution:{axis}:{value}"):
                    retained_tokens += int(row["stage_b_evidence"]["token_proxy_count"])
    for row in ranked:
        add(row, "objective_rank" if budget_applied else "retain_all_no_budget")

    remaining = [row for row in scored if row["chunk_uid"] not in selected_ids]
    baseline: List[Dict[str, Any]] = []
    baseline_tokens = 0
    for row in _stable_order(remaining, baseline_seed):
        count = int(row["stage_b_evidence"]["token_proxy_count"])
        if baseline_tokens + count <= selected_tokens:
            baseline.append({**row, "stage_b_baseline": {"arm": "stage_a_random_disjoint", "seed": baseline_seed}})
            baseline_tokens += count

    annotated_pool = annotate_retained_pool(
        scored,
        selected_ids=selected_ids,
        budget_applied=budget_applied,
    )
    annotated_by_id = {str(row["chunk_uid"]): row for row in annotated_pool}
    selected = [
        {
            **annotated_by_id[str(row["chunk_uid"])],
            "stage_b_selection": row["stage_b_selection"],
        }
        for row in selected
    ]
    budget_not_selected = [
        {
            **row,
            "stage_b_selection": {
                "selected": False,
                "accepted_by": None,
                "reason": BUDGET_NOT_SELECTED,
            },
        }
        for row in annotated_pool
        if (row.get("curation_decision") or {}).get("training_budget_disposition")
        == BUDGET_NOT_SELECTED
    ]

    return {
        "scored": annotated_pool,
        "curated_pool": annotated_pool,
        "selected": selected,
        "budget_not_selected": budget_not_selected,
        "baseline": baseline,
        "selection_mode": "budget_constrained" if budget_applied else "retain_all",
        "budget_applied": budget_applied,
        "full_curated_pool_token_proxy": total_tokens,
        "budget_token_proxy": budget,
        "selected_token_proxy": selected_tokens,
        "baseline_token_proxy": baseline_tokens,
        "disposition_summary": disposition_summary(annotated_pool),
        "invariants": {
            "stage_a_pass_implies_curated_pool_membership": True,
            "budget_not_selected_is_rejection": False,
            "retain_all_is_valid_outcome": True,
            "forced_rejection_quota": False,
        },
        "coverage_all": _coverage_counts(scored, coverage_axes),
        "coverage_selected": _coverage_counts(selected, coverage_axes),
        "coverage_baseline": _coverage_counts(baseline, coverage_axes),
        "coverage_tokens_all": source_token_counts,
        "coverage_tokens_selected": _coverage_token_counts(selected, distribution_axes or []),
        "coverage_tokens_baseline": _coverage_token_counts(baseline, distribution_axes or []),
    }
