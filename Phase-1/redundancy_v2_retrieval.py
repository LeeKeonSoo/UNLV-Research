from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations

from redundancy_v2 import RedundancySettings, RedundancyUnit, formatting_canonical, normalized_paragraphs, tokenize


@dataclass(frozen=True, slots=True)
class CandidatePair:
    left_uid: str
    right_uid: str
    retrieval_reasons: tuple[str, ...]


def _add_bucket_pairs(buckets: dict[bytes, list[str]], reason: str, pairs: dict[tuple[str, str], set[str]]) -> None:
    for members in buckets.values():
        for left, right in combinations(sorted(set(members)), 2):
            pairs.setdefault((left, right), set()).add(reason)


def _add_containment_anchor_pairs(
    window_index: dict[bytes, set[str]],
    anchors_by_uid: dict[str, tuple[bytes, bytes]],
    pairs: dict[tuple[str, str], set[str]],
) -> None:
    for uid, (first_anchor, last_anchor) in anchors_by_uid.items():
        candidates = window_index[first_anchor] & window_index[last_anchor]
        for candidate_uid in candidates - {uid}:
            left, right = sorted((uid, candidate_uid))
            pairs.setdefault((left, right), set()).add(
                "exact_containment_window_digest"
            )


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _shingles(text: str, size: int) -> tuple[bytes, ...]:
    tokens = tokenize(text)
    return tuple(
        hashlib.blake2b("\0".join(tokens[index : index + size]).encode("utf-8"), digest_size=8).digest()
        for index in range(max(0, len(tokens) - size + 1))
    )


def _window_digests(text: str, size: int) -> tuple[bytes, ...]:
    tokens = tokenize(text)
    return tuple(_digest("\0".join(tokens[index : index + size])) for index in range(max(0, len(tokens) - size + 1)))


def _minhash_signature(shingles: tuple[bytes, ...], size: int) -> tuple[int, ...]:
    return tuple(
        min(int.from_bytes(hashlib.blake2b(seed.to_bytes(2, "big") + shingle, digest_size=8).digest(), "big") for shingle in shingles)
        for seed in range(size)
    )


def retrieve_candidate_pairs(
    units: tuple[RedundancyUnit, ...],
    settings: RedundancySettings,
) -> tuple[CandidatePair, ...]:
    exact: dict[bytes, list[str]] = {}
    formatting: dict[bytes, list[str]] = {}
    paragraph: dict[bytes, list[str]] = {}
    containment_windows: dict[bytes, set[str]] = {}
    containment_anchors: dict[str, tuple[bytes, bytes]] = {}
    lsh: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    rows = settings.retrieval_signature_size // settings.retrieval_bands
    for unit in units:
        exact.setdefault(_digest(unit.text), []).append(unit.uid)
        formatting.setdefault(_digest(formatting_canonical(unit.text)), []).append(unit.uid)
        if settings.retrieve_repeated_span_candidates:
            for span in normalized_paragraphs(
                unit.text, settings.repeated_span_min_lexical_tokens
            ):
                paragraph.setdefault(_digest(span), []).append(unit.uid)
        windows = _window_digests(unit.text, settings.containment_min_tokens)
        if windows:
            containment_anchors[unit.uid] = (windows[0], windows[-1])
            for window in set(windows):
                containment_windows.setdefault(window, set()).add(unit.uid)
        if len(tokenize(unit.text)) < settings.retrieval_min_tokens:
            continue
        shingles = _shingles(unit.text, settings.retrieval_shingle_size)
        if not shingles:
            continue
        signature = _minhash_signature(shingles, settings.retrieval_signature_size)
        for band in range(settings.retrieval_bands):
            start = band * rows
            lsh.setdefault((band, signature[start : start + rows]), []).append(unit.uid)
    pairs: dict[tuple[str, str], set[str]] = {}
    _add_bucket_pairs(exact, "exact_digest", pairs)
    _add_bucket_pairs(formatting, "formatting_digest", pairs)
    if settings.retrieve_repeated_span_candidates:
        _add_bucket_pairs(paragraph, "repeated_paragraph_digest", pairs)
    _add_containment_anchor_pairs(containment_windows, containment_anchors, pairs)
    _add_bucket_pairs(lsh, "minhash_lsh", pairs)
    return tuple(CandidatePair(left, right, tuple(sorted(reasons))) for (left, right), reasons in sorted(pairs.items()))
