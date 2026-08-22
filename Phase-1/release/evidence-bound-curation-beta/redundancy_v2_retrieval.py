from __future__ import annotations

import hashlib
from dataclasses import dataclass

from redundancy_v2 import RedundancySettings, RedundancyUnit, formatting_canonical, normalized_paragraphs, tokenize


@dataclass(frozen=True, slots=True)
class CandidatePair:
    left_uid: str
    right_uid: str
    retrieval_reasons: tuple[str, ...]


def _add_bucket_pairs(buckets: dict[object, list[str]], reason: str, pairs: dict[tuple[str, str], set[str]]) -> None:
    for members in buckets.values():
        ordered = sorted(set(members))
        if len(ordered) < 2:
            continue
        representative = ordered[0]
        for member in ordered[1:]:
            pairs.setdefault((representative, member), set()).add(reason)


def _add_containment_anchor_pairs(
    window_index: dict[bytes, set[str]],
    anchors_by_uid: dict[str, tuple[bytes, bytes]],
    token_counts: dict[str, int],
    skip_uids: set[str],
    pairs: dict[tuple[str, str], set[str]],
) -> None:
    for uid, (first_anchor, last_anchor) in anchors_by_uid.items():
        if uid in skip_uids:
            continue
        candidates = (window_index[first_anchor] & window_index[last_anchor]) - {
            uid
        }
        if not candidates:
            continue
        candidate_uid = min(
            candidates, key=lambda item: (-token_counts[item], item)
        )
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


def _character_shingles(text: str, size: int) -> tuple[bytes, ...]:
    canonical = " ".join(formatting_canonical(text).split())
    return tuple(
        hashlib.blake2b(canonical[index : index + size].encode("utf-8"), digest_size=8).digest()
        for index in range(max(0, len(canonical) - size + 1))
    )


def _window_digests(text: str, size: int) -> tuple[bytes, ...]:
    tokens = tokenize(text)
    return tuple(_digest("\0".join(tokens[index : index + size])) for index in range(max(0, len(tokens) - size + 1)))


def _minhash_signature(shingles: tuple[bytes, ...], size: int) -> tuple[int, ...]:
    return tuple(
        min(int.from_bytes(hashlib.blake2b(seed.to_bytes(2, "big") + shingle, digest_size=8).digest(), "big") for shingle in shingles)
        for seed in range(size)
    )


def _mix64(value: int) -> int:
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _one_permutation_minhash_signature(
    shingles: tuple[bytes, ...], size: int
) -> tuple[int, ...]:
    """Build a densified one-permutation MinHash in O(shingles + size)."""
    empty = (1 << 64) - 1
    signature = [empty] * size
    for shingle in shingles:
        value = int.from_bytes(shingle, "big")
        bin_index = value % size
        bin_value = value // size
        if bin_value < signature[bin_index]:
            signature[bin_index] = bin_value
    populated = tuple(index for index, value in enumerate(signature) if value != empty)
    if not populated:
        return ()
    for index, value in enumerate(signature):
        if value != empty:
            continue
        donor = min(populated, key=lambda candidate: (candidate - index) % size)
        signature[index] = _mix64(signature[donor] ^ index)
    return tuple(signature)


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
    character_lsh: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    token_counts: dict[str, int] = {}
    rows = settings.retrieval_signature_size // settings.retrieval_bands
    for unit in units:
        tokens = tokenize(unit.text)
        token_counts[unit.uid] = len(tokens)
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
        if len(tokens) < settings.retrieval_min_tokens:
            continue
        shingles = _shingles(unit.text, settings.retrieval_shingle_size)
        if not shingles:
            continue
        signature = _minhash_signature(shingles, settings.retrieval_signature_size)
        for band in range(settings.retrieval_bands):
            start = band * rows
            lsh.setdefault((band, signature[start : start + rows]), []).append(unit.uid)
        character_shingles = _character_shingles(unit.text, settings.character_ngram_size)
        if character_shingles:
            character_signature = _one_permutation_minhash_signature(
                character_shingles,
                settings.character_minhash_bands * settings.character_minhashes_per_band,
            )
            for band in range(settings.character_minhash_bands):
                start = band * settings.character_minhashes_per_band
                character_lsh.setdefault(
                    (
                        band,
                        character_signature[
                            start : start + settings.character_minhashes_per_band
                        ],
                    ),
                    [],
                ).append(unit.uid)
    pairs: dict[tuple[str, str], set[str]] = {}
    _add_bucket_pairs(exact, "exact_digest", pairs)
    _add_bucket_pairs(formatting, "formatting_digest", pairs)
    if settings.retrieve_repeated_span_candidates:
        _add_bucket_pairs(paragraph, "repeated_paragraph_digest", pairs)
    identity_family_uids = {
        uid
        for buckets in (exact, formatting)
        for members in buckets.values()
        if len(set(members)) > 1
        for uid in members
    }
    _add_containment_anchor_pairs(
        containment_windows,
        containment_anchors,
        token_counts,
        identity_family_uids,
        pairs,
    )
    _add_bucket_pairs(lsh, "minhash_lsh", pairs)
    _add_bucket_pairs(character_lsh, "character_minhash_lsh", pairs)
    return tuple(CandidatePair(left, right, tuple(sorted(reasons))) for (left, right), reasons in sorted(pairs.items()))
