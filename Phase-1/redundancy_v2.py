from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"^\w+$", re.UNICODE)
CODE_SIGNAL_RE = re.compile(r"(?:\bdef\b|\bclass\b|\breturn\b|\bimport\b|=>|::|[{};])")
API_SIGNATURE_RE = re.compile(r"\b\w+\s*\([^)]*\)\s*(?:->|:)\s*[\w\[\], .]+")
NEGATION_TOKENS = frozenset({"no", "not", "never", "none", "without", "neither", "nor", "cannot"})
SUBSTANTIVE_OPERATORS = frozenset({"+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">=", "^", "%"})


class RedundancyContractError(RuntimeError):
    """Raised when a redundancy input violates the typed contract."""


class RelationType(str, Enum):
    EXACT_EQUIVALENT = "exact_equivalent"
    FORMATTING_EQUIVALENT = "formatting_equivalent"
    NEAR_SUBSTITUTE = "near_substitute"
    CONTAINED_PAYLOAD = "contained_payload"
    SUPERSET_PAYLOAD = "superset_payload"
    REPEATED_SPAN = "repeated_span"
    SEMANTIC_DUPLICATE_CANDIDATE = "semantic_duplicate_candidate"
    RELATED_COMPLEMENTARY = "related_complementary"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class RedundancySettings:
    short_exact_only_max_tokens: int = 32
    near_min_tokens: int = 64
    near_max_changed_ratio: float = 0.02
    near_max_changed_tokens: int = 4
    containment_min_tokens: int = 12
    repeated_span_min_lexical_tokens: int = 12
    complementary_overlap_floor: float = 0.18
    retrieval_min_tokens: int = 24
    retrieval_shingle_size: int = 5
    retrieval_signature_size: int = 32
    retrieval_bands: int = 8

    def __post_init__(self) -> None:
        if self.short_exact_only_max_tokens < 1 or self.near_min_tokens <= self.short_exact_only_max_tokens:
            raise RedundancyContractError("Near-duplicate length boundaries are inconsistent")
        if not 0.0 < self.near_max_changed_ratio < 1.0:
            raise RedundancyContractError("near_max_changed_ratio must be within (0, 1)")
        if self.retrieval_signature_size % self.retrieval_bands:
            raise RedundancyContractError("retrieval_signature_size must be divisible by retrieval_bands")


@dataclass(frozen=True, slots=True)
class RedundancyUnit:
    uid: str
    text: str

    def __post_init__(self) -> None:
        if not self.uid:
            raise RedundancyContractError("Redundancy unit identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    left_token_count: int
    right_token_count: int
    changed_left_count: int
    changed_right_count: int
    changed_ratio: float
    sequence_similarity: float
    left_containment: float
    right_containment: float
    substantive_difference_codes: tuple[str, ...]
    repeated_span_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RedundancyRelation:
    left_uid: str
    right_uid: str
    relation: RelationType
    reason_code: str
    evidence: RelationEvidence
    safe_family_edge: bool
    selection_authority: bool = False
    representative_selection_deferred: bool = True
    benchmark_outcomes_read: bool = False
    utility_read: bool = False


@dataclass(frozen=True, slots=True)
class RedundancyFamily:
    family_id: str
    member_uids: tuple[str, ...]
    edges: tuple[RedundancyRelation, ...]
    final_representative_uid: str | None = None
    representative_selection_deferred: bool = True


@dataclass(frozen=True, slots=True)
class RedundancyGraph:
    relations: tuple[RedundancyRelation, ...]
    families: tuple[RedundancyFamily, ...]


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text))


def formatting_canonical(text: str) -> str:
    """Normalize only line-ending encoding and one terminal newline."""
    line_normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return line_normalized.removesuffix("\n")


def normalized_paragraphs(text: str, minimum_lexical_tokens: int) -> tuple[str, ...]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = formatting_canonical(paragraph).strip("\n")
        lexical_count = sum(bool(WORD_RE.match(token)) for token in tokenize(normalized))
        if normalized and lexical_count >= minimum_lexical_tokens:
            paragraphs.append(normalized)
    return tuple(paragraphs)


def _changed_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    matcher = SequenceMatcher(a=left, b=right, autojunk=False)
    changed_left: list[str] = []
    changed_right: list[str] = []
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "equal":
            changed_left.extend(left[left_start:left_end])
            changed_right.extend(right[right_start:right_end])
    return tuple(changed_left), tuple(changed_right), matcher.ratio()


def _substantive_codes(left_text: str, right_text: str, changed: tuple[str, ...]) -> tuple[str, ...]:
    codes: set[str] = set()
    changed_words = [token for token in changed if WORD_RE.match(token)]
    if any(any(character.isdigit() for character in token) for token in changed):
        codes.add("numeric_constant_changed")
    if any(token in SUBSTANTIVE_OPERATORS for token in changed):
        codes.add("operator_changed")
    if any(token.casefold() in NEGATION_TOKENS for token in changed_words):
        codes.add("negation_changed")
    if any(len(token) == 1 and token in "ABCDE" for token in changed_words):
        codes.add("answer_label_changed")
    if (API_SIGNATURE_RE.search(left_text) or API_SIGNATURE_RE.search(right_text)) and changed:
        codes.add("api_signature_changed")
    if (CODE_SIGNAL_RE.search(left_text) or CODE_SIGNAL_RE.search(right_text)) and changed_words:
        codes.add("code_identifier_or_keyword_changed")
    if any(token[:1].isupper() and len(token) > 1 for token in changed_words):
        codes.add("named_entity_candidate_changed")
    return tuple(sorted(codes))


def _contains(sequence: tuple[str, ...], subsequence: tuple[str, ...]) -> bool:
    width = len(subsequence)
    return bool(width) and any(sequence[index : index + width] == subsequence for index in range(len(sequence) - width + 1))


def _relation_evidence(left: RedundancyUnit, right: RedundancyUnit, settings: RedundancySettings) -> RelationEvidence:
    left_tokens = tokenize(left.text)
    right_tokens = tokenize(right.text)
    changed_left, changed_right, similarity = _changed_tokens(left_tokens, right_tokens)
    shared = len(set(left_tokens) & set(right_tokens))
    repeated = sorted(
        set(normalized_paragraphs(left.text, settings.repeated_span_min_lexical_tokens))
        & set(normalized_paragraphs(right.text, settings.repeated_span_min_lexical_tokens))
    )
    denominator = max(1, min(len(left_tokens), len(right_tokens)))
    changed_ratio = max(len(changed_left), len(changed_right)) / denominator
    substantive = _substantive_codes(left.text, right.text, (*changed_left, *changed_right))
    return RelationEvidence(
        left_token_count=len(left_tokens),
        right_token_count=len(right_tokens),
        changed_left_count=len(changed_left),
        changed_right_count=len(changed_right),
        changed_ratio=changed_ratio,
        sequence_similarity=similarity,
        left_containment=shared / max(1, len(set(left_tokens))),
        right_containment=shared / max(1, len(set(right_tokens))),
        substantive_difference_codes=substantive,
        repeated_span_hashes=tuple(hashlib.sha256(span.encode("utf-8")).hexdigest() for span in repeated),
    )


def classify_relation(
    left: RedundancyUnit,
    right: RedundancyUnit,
    settings: RedundancySettings,
    *,
    semantic_candidate: bool = False,
) -> RedundancyRelation:
    evidence = _relation_evidence(left, right, settings)
    left_tokens = tokenize(left.text)
    right_tokens = tokenize(right.text)
    relation = RelationType.DISTINCT
    if left.text == right.text:
        relation = RelationType.EXACT_EQUIVALENT
    elif formatting_canonical(left.text) == formatting_canonical(right.text):
        relation = RelationType.FORMATTING_EQUIVALENT
    elif evidence.repeated_span_hashes:
        relation = RelationType.REPEATED_SPAN
    elif len(left_tokens) >= settings.containment_min_tokens and len(left_tokens) < len(right_tokens) and _contains(right_tokens, left_tokens):
        relation = RelationType.CONTAINED_PAYLOAD
    elif len(right_tokens) >= settings.containment_min_tokens and len(right_tokens) < len(left_tokens) and _contains(left_tokens, right_tokens):
        relation = RelationType.SUPERSET_PAYLOAD
    else:
        minimum_tokens = min(len(left_tokens), len(right_tokens))
        maximum_changes = max(evidence.changed_left_count, evidence.changed_right_count)
        near_boundary = max(1, min(settings.near_max_changed_tokens, int(minimum_tokens * settings.near_max_changed_ratio)))
        near = (
            minimum_tokens >= settings.near_min_tokens
            and maximum_changes <= near_boundary
            and evidence.changed_ratio <= settings.near_max_changed_ratio
            and not evidence.substantive_difference_codes
        )
        if near:
            relation = RelationType.NEAR_SUBSTITUTE
        elif semantic_candidate:
            relation = RelationType.SEMANTIC_DUPLICATE_CANDIDATE
        elif evidence.sequence_similarity >= settings.complementary_overlap_floor:
            relation = RelationType.RELATED_COMPLEMENTARY
    safe = relation in {RelationType.EXACT_EQUIVALENT, RelationType.FORMATTING_EQUIVALENT}
    return RedundancyRelation(
        left_uid=left.uid,
        right_uid=right.uid,
        relation=relation,
        reason_code=f"redundancy_{relation.value}",
        evidence=evidence,
        safe_family_edge=safe,
    )


def build_redundancy_graph(
    units: tuple[RedundancyUnit, ...],
    settings: RedundancySettings,
    *,
    exhaustive: bool = False,
) -> RedundancyGraph:
    from redundancy_v2_retrieval import retrieve_candidate_pairs

    pairs = tuple((left.uid, right.uid) for index, left in enumerate(units) for right in units[index + 1 :]) if exhaustive else tuple(
        (pair.left_uid, pair.right_uid) for pair in retrieve_candidate_pairs(units, settings)
    )
    by_uid = {unit.uid: unit for unit in units}
    relations = tuple(classify_relation(by_uid[left], by_uid[right], settings) for left, right in pairs)
    parents = {unit.uid: unit.uid for unit in units}

    def find(uid: str) -> str:
        while parents[uid] != uid:
            uid = parents[uid]
        return uid

    for edge in relations:
        if edge.safe_family_edge:
            left_root, right_root = find(edge.left_uid), find(edge.right_uid)
            parents[max(left_root, right_root)] = min(left_root, right_root)
    groups: dict[str, list[str]] = {}
    for uid in sorted(parents):
        groups.setdefault(find(uid), []).append(uid)
    families: list[RedundancyFamily] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        edges = tuple(edge for edge in relations if edge.safe_family_edge and edge.left_uid in member_set and edge.right_uid in member_set)
        family_id = hashlib.sha256("\0".join(members).encode("utf-8")).hexdigest()
        families.append(RedundancyFamily(family_id, tuple(members), edges))
    return RedundancyGraph(relations, tuple(sorted(families, key=lambda family: family.family_id)))
