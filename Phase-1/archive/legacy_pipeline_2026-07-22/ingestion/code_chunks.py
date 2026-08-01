"""Syntax-aware chunking and Stage-A hard gates for temporal code records."""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from collections import Counter
from typing import Any, Dict, Iterable, List

from ingestion.code_fingerprints import derived_fingerprints, simhash_hamming_distance


MIN_CODE_TOKENS = 5
MAX_DOCUMENTATION_WORDS = 200
HARD_NEAR_DUPLICATE_SIMHASH_DISTANCE = 3
PATHOLOGICAL_REPEATED_TOKEN_RUN = 20
PATHOLOGICAL_REPEATED_LINE_RATIO = 0.80
HARD_NEAR_DUPLICATE_JACCARD = 0.75
HARD_NEAR_DUPLICATE_CONTAINMENT = 0.88


class _SimhashBKIndex:
    def __init__(self) -> None:
        self.root: Dict[str, Any] | None = None

    def add(self, value: str | None, chunk_uid: str) -> None:
        if not value:
            return
        numeric = int(value, 16)
        if self.root is None:
            self.root = {"value": numeric, "chunk_uids": [chunk_uid], "children": {}}
            return
        node = self.root
        while True:
            distance = (numeric ^ int(node["value"])).bit_count()
            if distance == 0:
                node["chunk_uids"].append(chunk_uid)
                return
            children = node["children"]
            child = children.get(distance)
            if child is None:
                children[distance] = {"value": numeric, "chunk_uids": [chunk_uid], "children": {}}
                return
            node = child

    def within(self, value: str | None, maximum_distance: int) -> set[str]:
        if not value or self.root is None:
            return set()
        numeric = int(value, 16)
        result: set[str] = set()
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (numeric ^ int(node["value"])).bit_count()
            if distance <= maximum_distance:
                result.update(str(uid) for uid in node["chunk_uids"])
            low = distance - maximum_distance
            high = distance + maximum_distance
            pending.extend(
                child for edge, child in node["children"].items() if low <= int(edge) <= high
            )
        return result


def _nontrivia_python_tokens(text: str) -> List[str]:
    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    try:
        return [token.string for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type not in ignored]
    except (IndentationError, tokenize.TokenError):
        return []


def _line_span(text: str, start: int, end: int) -> str:
    lines = text.splitlines(keepends=True)
    return "".join(lines[max(0, start - 1) : max(0, end)]).strip()


def python_syntax_chunks(text: str) -> List[Dict[str, Any]]:
    tree = ast.parse(text)
    chunks: List[Dict[str, Any]] = []
    body = list(tree.body)
    first_statement_line = min((int(getattr(node, "lineno", 1)) for node in body), default=1)
    preamble = _line_span(text, 1, first_statement_line - 1)
    if preamble:
        chunks.append({"text": preamble, "kind": "module_preamble", "start_line": 1, "end_line": first_statement_line - 1})

    pending_simple: List[ast.stmt] = []

    def flush_simple() -> None:
        nonlocal pending_simple
        if not pending_simple:
            return
        start = int(pending_simple[0].lineno)
        end = int(pending_simple[-1].end_lineno or pending_simple[-1].lineno)
        value = _line_span(text, start, end)
        if value:
            chunks.append({"text": value, "kind": "module_statements", "start_line": start, "end_line": end})
        pending_simple = []

    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            flush_simple()
            start = min([int(node.lineno), *[int(item.lineno) for item in getattr(node, "decorator_list", [])]])
            end = int(node.end_lineno or node.lineno)
            chunks.append(
                {
                    "text": _line_span(text, start, end),
                    "kind": {
                        ast.FunctionDef: "function",
                        ast.AsyncFunctionDef: "async_function",
                        ast.ClassDef: "class",
                    }[type(node)],
                    "start_line": start,
                    "end_line": end,
                }
            )
        else:
            pending_simple.append(node)
    flush_simple()
    return [row for row in chunks if row["text"].strip()]


def documentation_chunks(text: str, maximum_words: int = MAX_DOCUMENTATION_WORDS) -> List[Dict[str, Any]]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    chunks: List[Dict[str, Any]] = []
    current: List[str] = []
    words = 0
    for paragraph in paragraphs or [text.strip()]:
        paragraph_words = len(paragraph.split())
        if current and words + paragraph_words > maximum_words:
            chunks.append({"text": "\n\n".join(current), "kind": "documentation_paragraph_group"})
            current, words = [], 0
        current.append(paragraph)
        words += paragraph_words
    if current:
        chunks.append({"text": "\n\n".join(current), "kind": "documentation_paragraph_group"})
    return chunks


def syntax_aware_chunks(record: Dict[str, Any]) -> Dict[str, Any]:
    text = str(record.get("text") or "")
    partition = record.get("partition") if isinstance(record.get("partition"), dict) else {}
    content_type = str(partition.get("content_type") or "")
    path = str(partition.get("path") or "")
    if path.lower().endswith(".py") and content_type in {"code", "test"}:
        try:
            chunks = python_syntax_chunks(text)
            return {"parseable": True, "parse_error": None, "chunks": chunks, "chunking_mode": "python_top_level_ast"}
        except SyntaxError as exc:
            return {
                "parseable": False,
                "parse_error": f"{exc.msg}:line={exc.lineno}:offset={exc.offset}",
                "chunks": [],
                "chunking_mode": "python_top_level_ast",
            }
    return {
        "parseable": True,
        "parse_error": None,
        "chunks": documentation_chunks(text),
        "chunking_mode": "documentation_paragraph_group",
    }


def _max_repeated_run(values: Iterable[str]) -> int:
    maximum = current = 0
    previous = None
    for value in values:
        if value == previous:
            current += 1
        else:
            previous, current = value, 1
        maximum = max(maximum, current)
    return maximum


def token_shingles(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < 2:
        return set(tokens)
    width = 1 if len(tokens) < 10 else (2 if len(tokens) < 24 else 3)
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def _hard_overlap(left: set[str], right: set[str]) -> Dict[str, float]:
    if not left or not right:
        return {"jaccard": 0.0, "containment": 0.0}
    intersection = len(left.intersection(right))
    return {
        "jaccard": intersection / len(left.union(right)),
        "containment": intersection / min(len(left), len(right)),
    }


def hard_near_duplicate_evidence(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    left_simhash = left.get("token_simhash64")
    right_simhash = right.get("token_simhash64")
    if not left_simhash or not right_simhash:
        return {"match": False, "simhash_distance": None, "jaccard": 0.0, "containment": 0.0}
    distance = simhash_hamming_distance(str(left_simhash), str(right_simhash))
    if distance > HARD_NEAR_DUPLICATE_SIMHASH_DISTANCE:
        return {"match": False, "simhash_distance": distance, "jaccard": 0.0, "containment": 0.0}
    overlap = _hard_overlap(left["shingles"], right["shingles"])
    return {
        "match": (
            overlap["jaccard"] >= HARD_NEAR_DUPLICATE_JACCARD
            or overlap["containment"] >= HARD_NEAR_DUPLICATE_CONTAINMENT
        ),
        "simhash_distance": distance,
        **overlap,
    }


def stage_a_local_evidence(chunk: Dict[str, Any]) -> Dict[str, Any]:
    text = str(chunk["text"])
    path = str(chunk.get("path") or "")
    is_python = path.lower().endswith(".py")
    python_tokens = _nontrivia_python_tokens(text) if is_python else []
    parseable = True
    parse_error = None
    if is_python:
        try:
            ast.parse(text)
        except SyntaxError as exc:
            parseable = False
            parse_error = f"{exc.msg}:line={exc.lineno}:offset={exc.offset}"
    lexical_tokens = re.findall(r"[a-z_]\w*|\d+(?:\.\d+)?", text.lower())
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_counts = Counter(nonempty_lines)
    repeated_line_ratio = (
        max(line_counts.values()) / len(nonempty_lines) if nonempty_lines else 0.0
    )
    minimum_unit = len(python_tokens) >= MIN_CODE_TOKENS if is_python else len(text.split()) >= 20
    pathological = (
        _max_repeated_run(lexical_tokens) >= PATHOLOGICAL_REPEATED_TOKEN_RUN
        or (len(nonempty_lines) >= 5 and repeated_line_ratio >= PATHOLOGICAL_REPEATED_LINE_RATIO)
    )
    fingerprints = derived_fingerprints(text)
    canonical_payload = (
        ast.dump(ast.parse(text), annotate_fields=True, include_attributes=False)
        if is_python and parseable
        else re.sub(r"\s+", " ", text).strip()
    )
    return {
        "parseable": parseable,
        "parse_error": parse_error,
        "minimum_learnable_unit": minimum_unit,
        "pathological_repetition": pathological,
        "max_repeated_token_run": _max_repeated_run(lexical_tokens),
        "repeated_line_ratio": round(repeated_line_ratio, 6),
        "text_sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "canonical_content_sha256": hashlib.sha256(
            canonical_payload.encode("utf-8", errors="replace")
        ).hexdigest(),
        **fingerprints,
    }


def _stage_a_local_blockers(local: Dict[str, Any]) -> List[str]:
    blockers = []
    if not local["parseable"]:
        blockers.append("python_chunk_not_parseable")
    if not local["minimum_learnable_unit"]:
        blockers.append("below_minimum_learnable_unit")
    if local["pathological_repetition"]:
        blockers.append("pathological_repetition")
    return blockers


def apply_stage_a_hard_gates(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = [stage_a_local_evidence(chunk) for chunk in chunks]
    chunk_uids = [str(chunk["chunk_uid"]) for chunk in chunks]
    if len(set(chunk_uids)) != len(chunk_uids):
        raise ValueError("Stage-A chunk_uid values must be unique for deterministic duplicate lineage.")

    prepared = [
        {
            "index": index,
            "chunk": chunk,
            "chunk_uid": str(chunk["chunk_uid"]),
            "local": local,
            "local_blockers": _stage_a_local_blockers(local),
            "shingles": token_shingles(str(chunk.get("text") or "")),
        }
        for index, (chunk, local) in enumerate(zip(chunks, evidence))
    ]
    duplicate_evidence: Dict[int, Dict[str, Any]] = {
        row["index"]: {
            "exact_duplicate_match": None,
            "hard_near_duplicate_match": None,
            "hard_near_duplicate_overlap": None,
        }
        for row in prepared
    }

    accepted_representatives: List[Dict[str, Any]] = []
    exact_representatives: Dict[str, Dict[str, Any]] = {}
    near_duplicate_index = _SimhashBKIndex()
    eligible = sorted(
        (row for row in prepared if not row["local_blockers"]),
        key=lambda row: (row["chunk_uid"], row["local"]["text_sha256"]),
    )
    for row in eligible:
        exact = str(row["local"]["canonical_content_sha256"])
        exact_representative = exact_representatives.get(exact)
        if exact_representative is not None:
            duplicate_evidence[row["index"]]["exact_duplicate_match"] = exact_representative["chunk_uid"]
            continue

        near_match = None
        near_overlap = None
        candidate = {**row["local"], "shingles": row["shingles"]}
        near_candidate_uids = near_duplicate_index.within(
            candidate.get("token_simhash64"), HARD_NEAR_DUPLICATE_SIMHASH_DISTANCE
        )
        for representative in accepted_representatives:
            if representative["chunk_uid"] not in near_candidate_uids:
                continue
            overlap = hard_near_duplicate_evidence(candidate, representative)
            if overlap["match"]:
                near_match = representative["chunk_uid"]
                near_overlap = {
                    name: round(value, 6) if isinstance(value, float) else value
                    for name, value in overlap.items()
                }
                break
        if near_match is not None:
            duplicate_evidence[row["index"]]["hard_near_duplicate_match"] = near_match
            duplicate_evidence[row["index"]]["hard_near_duplicate_overlap"] = near_overlap

        representative = {
            "chunk_uid": row["chunk_uid"],
            "shingles": row["shingles"],
            **row["local"],
        }
        accepted_representatives.append(representative)
        near_duplicate_index.add(representative.get("token_simhash64"), representative["chunk_uid"])
        exact_representatives[exact] = representative

    decisions = []
    for row in prepared:
        duplicate = duplicate_evidence[row["index"]]
        blockers = list(row["local_blockers"])
        if duplicate["exact_duplicate_match"] is not None:
            blockers.append("exact_duplicate_within_split")
        if duplicate["hard_near_duplicate_match"] is not None:
            blockers.append("hard_near_duplicate_within_split")
        decisions.append(
            {
                **row["chunk"],
                "stage_a_evidence": row["local"],
                "duplicate_representative_eligible": not row["local_blockers"],
                "duplicate_representative_policy": "local_gate_pass_then_canonical_exact_lexicographic_v2",
                **duplicate,
                "stage_a_blockers": sorted(set(blockers)),
                "stage_a_pass": not blockers,
            }
        )
    return decisions
