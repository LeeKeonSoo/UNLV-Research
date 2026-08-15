from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonRow = Mapping[str, JsonValue]
MutableJsonRow = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class CalibrationSampleConfig:
    target_size: int
    seed: str

    def __post_init__(self) -> None:
        if self.target_size < 1 or not self.seed:
            raise ValueError("Calibration sampling requires a positive size and frozen seed")


def _first_label(row: JsonRow, key: str, fallback: str) -> str:
    value = row.get(key)
    if not isinstance(value, list) or not value:
        return fallback
    first = value[0]
    return first if isinstance(first, str) and first else fallback


def _uid(row: JsonRow) -> str:
    value = row.get("uid") or row.get("chunk_uid")
    if not isinstance(value, str) or not value:
        raise ValueError("Calibration rows require uid or chunk_uid")
    return value


def _length_bin(row: JsonRow) -> str:
    value = row.get("token_proxy")
    tokens = value if isinstance(value, int) else 0
    if tokens <= 32:
        return "short"
    if tokens <= 256:
        return "medium"
    return "long"


def _stable_score(seed: str, uid: str) -> str:
    return hashlib.sha256(f"{seed}\0{uid}".encode()).hexdigest()


def normalized_text_sha256(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_calibration_rows(
    rows: Sequence[JsonRow],
    config: CalibrationSampleConfig,
) -> tuple[MutableJsonRow, ...]:
    """Select a deterministic round-robin sample across observable text strata."""
    if config.target_size > len(rows):
        raise ValueError("Calibration sample cannot exceed the input corpus")
    unique_rows: list[JsonRow] = []
    observed_hashes: set[str] = set()
    for row in sorted(rows, key=lambda item: _stable_score(config.seed, _uid(item))):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Calibration rows require non-empty text")
        text_hash = normalized_text_sha256(text)
        if text_hash in observed_hashes:
            continue
        observed_hashes.add(text_hash)
        unique_rows.append(row)
    if config.target_size > len(unique_rows):
        raise ValueError("Calibration sample cannot exceed unique normalized payloads")
    groups: dict[tuple[str, str, str, str], list[JsonRow]] = defaultdict(list)
    for row in unique_rows:
        stratum = (
            _first_label(row, "route_labels", "unknown"),
            _first_label(row, "script_labels", "unknown"),
            _first_label(row, "format_labels", "unknown"),
            _length_bin(row),
        )
        groups[stratum].append(row)
    ordered = {
        key: sorted(group, key=lambda row: _stable_score(config.seed, _uid(row)))
        for key, group in groups.items()
    }
    selected: list[tuple[JsonRow, tuple[str, str, str, str]]] = []
    positions = {key: 0 for key in ordered}
    while len(selected) < config.target_size:
        progressed = False
        for key in sorted(ordered):
            position = positions[key]
            if position >= len(ordered[key]):
                continue
            selected.append((ordered[key][position], key))
            positions[key] += 1
            progressed = True
            if len(selected) == config.target_size:
                break
        if not progressed:
            raise RuntimeError("Calibration sampler exhausted the corpus unexpectedly")
    annotated: list[MutableJsonRow] = []
    for row, (route, script, format_label, length_bin) in selected:
        output = dict(row)
        output["chunk_uid"] = _uid(row)
        output["quality_calibration_sample"] = True
        output["quality_calibration_stratum"] = {
            "route": route,
            "script": script,
            "format": format_label,
            "length_bin": length_bin,
        }
        annotated.append(output)
    return tuple(sorted(annotated, key=lambda row: _stable_score(config.seed, _uid(row))))


def select_protected_rows(
    rows: Sequence[JsonRow],
    *,
    calibration_rows: Sequence[JsonRow],
    config: CalibrationSampleConfig,
) -> tuple[MutableJsonRow, ...]:
    calibration_uids = {_uid(row) for row in calibration_rows}
    calibration_hashes = {
        normalized_text_sha256(str(row["text"])) for row in calibration_rows
    }
    eligible = tuple(
        row
        for row in rows
        if _uid(row) not in calibration_uids
        and normalized_text_sha256(str(row.get("text") or "")) not in calibration_hashes
    )
    selected = select_calibration_rows(eligible, config)
    protected: list[MutableJsonRow] = []
    for row in selected:
        output = dict(row)
        stratum = output.pop("quality_calibration_stratum")
        output.pop("quality_calibration_sample")
        output["quality_protected_sample"] = True
        output["quality_protected_stratum"] = stratum
        protected.append(output)
    return tuple(protected)
