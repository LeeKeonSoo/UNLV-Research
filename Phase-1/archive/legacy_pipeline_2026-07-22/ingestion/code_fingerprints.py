"""Derived code fingerprints for benchmark near-duplicate quarantine."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any, Dict, Iterable


TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s\w]", re.UNICODE)


class _StructuralNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="_name", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg="_arg", annotation=self.visit(node.annotation) if node.annotation else None), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        value: Any = node.value
        if isinstance(value, str):
            value = "_str"
        elif isinstance(value, bytes):
            value = b"_bytes"
        elif isinstance(value, (int, float, complex)):
            value = 0
        return ast.copy_location(ast.Constant(value=value), node)


def _token_shingles(text: str, width: int = 5) -> Iterable[str]:
    tokens = [token.lower() for token in TOKEN_RE.findall(str(text or ""))]
    for index in range(max(0, len(tokens) - width + 1)):
        yield "\x1f".join(tokens[index : index + width])


def token_simhash64(text: str) -> str | None:
    vector = [0] * 64
    count = 0
    for shingle in _token_shingles(text):
        count += 1
        value = int.from_bytes(hashlib.sha256(shingle.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    if count == 0:
        return None
    result = sum((1 << bit) for bit, score in enumerate(vector) if score >= 0)
    return f"{result:016x}"


def python_ast_sha256(text: str) -> str | None:
    try:
        tree = ast.parse(str(text or ""))
    except (SyntaxError, ValueError):
        return None
    normalized = _StructuralNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    payload = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derived_fingerprints(text: str) -> Dict[str, str]:
    result = {}
    simhash = token_simhash64(text)
    if simhash:
        result["token_simhash64"] = simhash
    ast_hash = python_ast_sha256(text)
    if ast_hash:
        result["python_ast_sha256"] = ast_hash
    return result


def simhash_hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()
