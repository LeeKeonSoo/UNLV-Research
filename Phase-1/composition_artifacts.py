from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from content_router import ROUTE_ORDER, route_content


PRIMARY_ROUTE_ORDER: Final = tuple(
    label for label in ROUTE_ORDER if label not in {"mixed", "unknown"}
) + ("mixed", "unknown")


class CompositionArtifactError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CompositionRecord:
    uid: str
    text: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.uid or self.token_count < 0:
            raise CompositionArtifactError(
                "Composition records require an ID and nonnegative token count"
            )


@dataclass(frozen=True, slots=True)
class CompositionShare:
    stage: str
    axis: str
    label: str
    record_count: int
    token_count: int
    token_share: float


@dataclass(frozen=True, slots=True)
class CompositionDelta:
    axis: str
    label: str
    raw_token_count: int
    curated_token_count: int
    token_count_delta: int
    token_share_delta: float


@dataclass(frozen=True, slots=True)
class CompositionAuditArtifacts:
    shares: tuple[CompositionShare, ...]
    deltas: tuple[CompositionDelta, ...]
    raw_tokens: int
    curated_tokens: int
    authority: str = "audit_only"
    consumed_by_selection: bool = False
    target_distribution_enforced: bool = False


@dataclass(frozen=True, slots=True)
class CompositionArtifactPaths:
    audit_json: Path
    route_csv: Path
    language_csv: Path
    delta_csv: Path

    def all(self) -> tuple[Path, ...]:
        return (self.audit_json, self.route_csv, self.language_csv, self.delta_csv)


def _primary_route(labels: list[str]) -> str:
    if "mixed" in labels:
        return "mixed"
    for candidate in PRIMARY_ROUTE_ORDER:
        if candidate in labels:
            return candidate
    return "unknown"


def _labels(record: CompositionRecord) -> dict[str, tuple[str, ...]]:
    routing = route_content(record.text)
    routes = tuple(routing["route_labels"])
    scripts = tuple(routing["language_script"]["labels"])
    return {
        "primary_route": (_primary_route(routing["route_labels"]),),
        "route_incidence": routes,
        "language_script_incidence": scripts,
    }


def _stage_shares(
    stage: str, records: tuple[CompositionRecord, ...]
) -> tuple[CompositionShare, ...]:
    total_tokens = sum(record.token_count for record in records)
    record_counts: Counter[tuple[str, str]] = Counter()
    token_counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        for axis, labels in _labels(record).items():
            for label in labels:
                record_counts[(axis, label)] += 1
                token_counts[(axis, label)] += record.token_count
    return tuple(
        CompositionShare(
            stage,
            axis,
            label,
            record_counts[(axis, label)],
            token_count,
            token_count / total_tokens if total_tokens else 0.0,
        )
        for (axis, label), token_count in sorted(token_counts.items())
    )


def build_composition_artifacts(
    raw_records: tuple[CompositionRecord, ...],
    curated_records: tuple[CompositionRecord, ...],
) -> CompositionAuditArtifacts:
    if len({record.uid for record in raw_records}) != len(raw_records):
        raise CompositionArtifactError("Raw composition record IDs must be unique")
    if len({record.uid for record in curated_records}) != len(curated_records):
        raise CompositionArtifactError("Curated composition record IDs must be unique")
    shares = _stage_shares("raw", raw_records) + _stage_shares(
        "curated", curated_records
    )
    by_key = {(item.stage, item.axis, item.label): item for item in shares}
    axes_and_labels = {
        (item.axis, item.label) for item in shares
    }
    deltas = tuple(
        CompositionDelta(
            axis,
            label,
            by_key.get(
                ("raw", axis, label), CompositionShare("raw", axis, label, 0, 0, 0.0)
            ).token_count,
            by_key.get(
                ("curated", axis, label),
                CompositionShare("curated", axis, label, 0, 0, 0.0),
            ).token_count,
            by_key.get(
                ("curated", axis, label),
                CompositionShare("curated", axis, label, 0, 0, 0.0),
            ).token_count
            - by_key.get(
                ("raw", axis, label), CompositionShare("raw", axis, label, 0, 0, 0.0)
            ).token_count,
            by_key.get(
                ("curated", axis, label),
                CompositionShare("curated", axis, label, 0, 0, 0.0),
            ).token_share
            - by_key.get(
                ("raw", axis, label), CompositionShare("raw", axis, label, 0, 0, 0.0)
            ).token_share,
        )
        for axis, label in sorted(axes_and_labels)
    )
    return CompositionAuditArtifacts(
        shares,
        deltas,
        sum(record.token_count for record in raw_records),
        sum(record.token_count for record in curated_records),
    )


def _write_shares(
    path: Path, shares: tuple[CompositionShare, ...], axes: frozenset[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("stage", "axis", "label", "record_count", "token_count", "token_share"),
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in shares if item.axis in axes)


def write_composition_artifacts(
    audit: CompositionAuditArtifacts, output_directory: Path
) -> CompositionArtifactPaths:
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = CompositionArtifactPaths(
        output_directory / "composition_audit.json",
        output_directory / "composition_by_route.csv",
        output_directory / "composition_by_language.csv",
        output_directory / "raw_curated_composition_delta.csv",
    )
    payload = {
        "authority": audit.authority,
        "consumed_by_selection": audit.consumed_by_selection,
        "target_distribution_enforced": audit.target_distribution_enforced,
        "raw_tokens": audit.raw_tokens,
        "curated_tokens": audit.curated_tokens,
        "shares": [asdict(item) for item in audit.shares],
        "deltas": [asdict(item) for item in audit.deltas],
    }
    paths.audit_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_shares(
        paths.route_csv, audit.shares, frozenset({"primary_route", "route_incidence"})
    )
    _write_shares(
        paths.language_csv, audit.shares, frozenset({"language_script_incidence"})
    )
    with paths.delta_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "axis",
                "label",
                "raw_token_count",
                "curated_token_count",
                "token_count_delta",
                "token_share_delta",
            ),
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in audit.deltas)
    return paths
